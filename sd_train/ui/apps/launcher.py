import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, ProgressBar, Static, TextArea
from textual.widgets.option_list import Option
from textual_autocomplete import DropdownItem, PathAutoComplete, TargetState

from sd_train.config.models import DEFAULT_LOCAL_ENVIRONMENT_NAME
from sd_train.tagger.core import (
    TaggerModelConfig,
    TaggerRunSummary,
    TagStatsSnapshot,
    add_tags,
    auto_tag,
    collect_stats,
    count_overwrite_candidates,
    delete_all_tags,
    front_tags,
    parse_tag_input,
    rename_single_tag,
    remove_single_tag,
    shuffle_tags,
)


@dataclass
class RunSelection:
    environment_name: str
    train_config_path: str
    train_script: str


@dataclass
class LauncherResult:
    action: Literal["start", "quit", "tagger"]
    environments: list[dict[str, Any]]
    last: dict[str, str]
    other_options: dict[str, str]
    selection: RunSelection | None


@dataclass
class TaggerWorkspaceResult:
    action: Literal["back", "quit"]
    dataset_dir: str
    model: str
    threshold: float
    batch: int


@dataclass
class EnvironmentPickerResult:
    environments: list[dict[str, Any]]
    selected_name: str | None


class PreflightReviewApp(App[bool]):
    CSS = """
    #root {
        height: 1fr;
        border: solid #666666;
        padding: 1;
    }

    #title {
        height: auto;
        text-style: bold;
        margin-bottom: 1;
    }

    #body {
        height: 1fr;
        overflow-y: auto;
    }

    #actions {
        height: auto;
        margin-top: 1;
    }

    #actions > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #actions > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #actions.start-focus > .option-list--option-highlighted {
        background: #166534;
    }

    #actions.danger-focus > .option-list--option-highlighted {
        background: #b42318;
    }
    """

    BINDINGS = [
        Binding("enter", "activate_selected", "Select"),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, summary: str, error_message: str | None = None):
        super().__init__()
        self.summary = summary
        self.error_message = error_message

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            title = "Preflight Validation Failed" if self.error_message else "Preflight Summary"
            yield Static(title, id="title")
            body = self.error_message if self.error_message is not None else self.summary
            yield Static(body, id="body")
            yield OptionList(id="actions")

    def on_mount(self) -> None:
        actions = self.query_one("#actions", OptionList)
        if self.error_message is None:
            actions.add_option(Option(_start_label("Proceed"), id="action:proceed"))
            actions.add_option(Option(_danger_label("Cancel"), id="action:cancel"))
        else:
            actions.add_option(Option(_danger_label("Close"), id="action:cancel"))
        actions.focus()

    def action_activate_selected(self) -> None:
        actions = self.query_one("#actions", OptionList)
        index = actions.highlighted
        if index is None:
            return
        option = actions.get_option_at_index(index)
        self._apply_action(option.id)

    def action_cancel(self) -> None:
        self.exit(False)

    def _apply_action(self, option_id: str | None) -> None:
        if option_id == "action:proceed":
            self.exit(True)
            return
        self.exit(False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        actions = self.query_one("#actions", OptionList)
        if event.option_list is not actions:
            return
        self._apply_action(event.option_id)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        actions = self.query_one("#actions", OptionList)
        if event.option_list is not actions:
            return
        if event.option_id == "action:proceed":
            actions.add_class("start-focus")
            actions.remove_class("danger-focus")
            return
        actions.remove_class("start-focus")
        actions.add_class("danger-focus")


def _action_label(text: str) -> Text:
    return Text(text, style="bold #dbeafe")


def _danger_label(text: str) -> Text:
    return Text(text, style="bold #ff8f8f")


def _start_label(text: str) -> Text:
    return Text(text, style="bold #86efac")


def _field_label(label: str, value: str) -> Text:
    text = Text()
    text.append(f"{label}: ", style="bold")
    text.append(value, style="#a6adbb")
    return text


class EditValueScreen(ModalScreen[str | None]):
    CSS = """
    EditValueScreen {
        align: center middle;
    }

    #dialog {
        width: 64;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    .row {
        height: auto;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, initial_value: str, password: bool = False):
        super().__init__()
        self.dialog_title = title
        self.initial_value = initial_value
        self.password = password

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.dialog_title)
            yield Input(value=self.initial_value, id="value", password=self.password)
            yield Static("Enter=apply, Esc=cancel", id="hints")

    def on_mount(self) -> None:
        self.query_one("#value", Input).focus()

    def action_save(self) -> None:
        self.dismiss(self.query_one("#value", Input).value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self.action_save()


class PathEditScreen(ModalScreen[str | None]):
    CSS = """
    PathEditScreen {
        align: center middle;
    }

    #dialog {
        width: 72;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #value {
        margin-top: 1;
        margin-bottom: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, initial_value: str, directories_only: bool = False):
        super().__init__()
        self.dialog_title = title
        self.initial_value = initial_value
        self.directories_only = directories_only

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.dialog_title)
            yield Input(value=self.initial_value, id="value")
            if self.directories_only:
                yield DirectoryPathAutoComplete(
                    "#value",
                    path=".",
                    prevent_default_enter=False,
                    id="path_autocomplete",
                )
            else:
                yield PathAutoComplete(
                    "#value",
                    path=".",
                    prevent_default_enter=False,
                    id="path_autocomplete",
                )
            yield Static("Enter=apply, Esc=cancel", id="hints")

    def on_mount(self) -> None:
        self.query_one("#value", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "value":
            return
        self.dismiss(event.value.strip())


class DirectoryPathAutoComplete(PathAutoComplete):
    def get_candidates(self, target_state: TargetState) -> list[DropdownItem]:
        current_input = target_state.text[: target_state.cursor_position]

        if "/" in current_input:
            last_slash_index = current_input.rindex("/")
            path_segment = current_input[:last_slash_index] or "/"
            directory = self.path / path_segment if path_segment != "/" else self.path
        else:
            directory = self.path

        try:
            entries = list(os.scandir(directory))
        except OSError:
            return []

        results: list[DropdownItem] = []
        for entry in entries:
            if not entry.is_dir():
                continue
            completion = entry.name
            if not self.show_dotfiles and completion.startswith("."):
                continue
            completion += "/"
            results.append(DropdownItem(completion, prefix=self.folder_prefix))

        results.sort(key=lambda item: str(item.main).lower())
        return results


class EditMultilineScreen(ModalScreen[str | None]):
    CSS = """
    EditMultilineScreen {
        align: center middle;
    }

    #dialog {
        width: 80;
        height: 80%;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #text {
        height: 1fr;
        margin-top: 1;
    }

    .row {
        height: auto;
        margin-top: 1;
    }

    #actions {
        height: auto;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    #actions > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #actions > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, initial_value: str):
        super().__init__()
        self.dialog_title = title
        self.initial_value = initial_value

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.dialog_title)
            yield TextArea(self.initial_value, id="text")
            yield OptionList(
                Option(_action_label("Save"), id="action:save"),
                Option(_action_label("Cancel"), id="action:cancel"),
                id="actions",
            )
            yield Static("Esc=cancel", id="hints")

    def on_mount(self) -> None:
        self.query_one("#text", TextArea).focus()

    def on_text_area_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.action_cancel()
            event.stop()

    def action_save(self) -> None:
        self.dismiss(self.query_one("#text", TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        actions = self.query_one("#actions", OptionList)
        if event.option_list is not actions:
            return
        if event.option_id == "action:save":
            self.action_save()
        else:
            self.action_cancel()


class SelectValueScreen(ModalScreen[str | None]):
    CSS = """
    SelectValueScreen {
        align: center middle;
    }

    #dialog {
        width: 80;
        height: 80%;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #items {
        height: 1fr;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, options: list[str], selected: str):
        super().__init__()
        self.dialog_title = title
        self.options = options
        self.selected = selected

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.dialog_title)
            yield OptionList(id="items")
            yield Static("Enter=apply, Esc=cancel", id="hints")

    def on_mount(self) -> None:
        items = self.query_one("#items", OptionList)
        for value in self.options:
            marker = " (selected)" if value == self.selected else ""
            items.add_option(Option(value + marker, id=f"value:{value}"))
        items.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option_id
        if option_id is None or not option_id.startswith("value:"):
            return
        self.dismiss(option_id.removeprefix("value:"))


class ConfirmDeleteScreen(ModalScreen[bool]):
    CSS = """
    ConfirmDeleteScreen {
        align: center middle;
    }

    #dialog {
        width: 64;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #options {
        height: auto;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    #options > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #options > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #options.danger-focus > .option-list--option-highlighted {
        background: #b42318;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, env_name: str):
        super().__init__()
        self.env_name = env_name

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(f"Delete environment '{self.env_name}'?")
            yield OptionList(
                Option(_danger_label("Delete"), id="danger:delete"),
                Option(_action_label("Cancel"), id="action:cancel"),
                id="options",
            )
            yield Static("Enter=select, Esc=cancel", id="hints")

    def on_mount(self) -> None:
        self.query_one("#options", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        options = self.query_one("#options", OptionList)
        if event.option_list is not options:
            return
        self.dismiss(event.option_id == "danger:delete")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        options = self.query_one("#options", OptionList)
        if event.option_list is not options:
            return
        if event.option_id == "danger:delete":
            options.add_class("danger-focus")
        else:
            options.remove_class("danger-focus")


class SelectEnvironmentScreen(ModalScreen[EnvironmentPickerResult]):
    CSS = """
    SelectEnvironmentScreen {
        align: center middle;
    }

    #dialog {
        width: 70;
        height: 70%;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #env_options {
        height: 1fr;
        margin-top: 1;
    }

    #env_options > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #env_options > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    """

    BINDINGS = [
        Binding("enter", "select_current", "Select", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("E", "edit_selected", "Edit", show=False),
        Binding("X", "delete_selected", "Delete", show=False),
    ]

    def __init__(
        self,
        environments: list[dict[str, Any]],
        selected: str,
        default_offer_query: str,
    ):
        super().__init__()
        self.environments = [dict(env) for env in environments]
        self.selected = selected
        self.default_offer_query = default_offer_query
        self._pending_delete_name: str | None = None

    def _is_builtin_local_name(self, name: str) -> bool:
        return name == DEFAULT_LOCAL_ENVIRONMENT_NAME

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Select Environment")
            yield OptionList(id="env_options")
            yield Static(
                "Enter=select, Shift+E=edit, Shift+X=delete, Esc=cancel", id="hints"
            )

    def on_mount(self) -> None:
        self._render_options()
        option_list = self.query_one("#env_options", OptionList)
        names = self._env_names()
        if self.selected and self.selected in names:
            option_list.highlighted = names.index(self.selected)
        elif not names:
            option_list.highlighted = 2
        option_list.focus()

    def _env_names(self) -> list[str]:
        return [
            str(env.get("name", "")) for env in self.environments if env.get("name")
        ]

    def _render_options(self) -> None:
        env_options = self.query_one("#env_options", OptionList)
        env_options.clear_options()
        names = self._env_names()
        if not names:
            env_options.add_option(
                Option("(no environments)", id="env:none", disabled=True)
            )
        else:
            for name in names:
                env_options.add_option(Option(name, id=f"env:{name}"))
        env_options.add_option(Option(" ", id="divider:actions", disabled=True))
        env_options.add_option(
            Option(_action_label("Add New Environment"), id="action:add")
        )

    def _selected_option_id(self) -> str | None:
        env_options = self.query_one("#env_options", OptionList)
        index = env_options.highlighted
        if index is None:
            return None
        return env_options.get_option_at_index(index).id

    def action_select_current(self) -> None:
        option_id = self._selected_option_id()
        if option_id is None:
            return
        if option_id == "env:none":
            return
        if option_id == "action:add":
            self._open_editor(is_new=True, name=None)
            return
        if option_id.startswith("env:"):
            self.dismiss(
                EnvironmentPickerResult(
                    environments=self.environments,
                    selected_name=option_id[4:],
                )
            )

    def action_cancel(self) -> None:
        self.dismiss(
            EnvironmentPickerResult(
                environments=self.environments,
                selected_name=None,
            )
        )

    def action_edit_selected(self) -> None:
        option_id = self._selected_option_id()
        if option_id is None or not option_id.startswith("env:"):
            return
        name = option_id[4:]
        if self._is_builtin_local_name(name):
            return
        self._open_editor(is_new=False, name=name)

    def action_delete_selected(self) -> None:
        option_id = self._selected_option_id()
        if option_id is None or not option_id.startswith("env:"):
            return
        name = option_id[4:]
        if self._is_builtin_local_name(name):
            return
        self._pending_delete_name = name
        self.app.push_screen(ConfirmDeleteScreen(name), self._on_confirm_delete)

    def _default_env(self) -> dict[str, Any]:
        return {
            "name": "",
            "type": "ssh",
            "host": "",
            "user": "ubuntu",
            "port": 22,
            "identity_file": "",
            "api_key": "",
            "offer_query": self.default_offer_query,
            "order": "dph",
            "disk": 50,
        }

    def _open_editor(self, is_new: bool, name: str | None) -> None:
        if is_new:
            env = self._default_env()
        else:
            env = next(
                (
                    it
                    for it in self.environments
                    if str(it.get("name", "")) == (name or "")
                ),
                None,
            )
            if env is None:
                return
            env = dict(env)
        self.app.push_screen(
            EnvironmentEditScreen(
                env,
                is_new=is_new,
                default_offer_query=self.default_offer_query,
            ),
            self._on_editor_closed,
        )

    def _on_editor_closed(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        if result.get("__delete__") is True:
            name = str(result.get("name", "")).strip()
            if not name or self._is_builtin_local_name(name):
                return
            self._pending_delete_name = name
            self.app.push_screen(ConfirmDeleteScreen(name), self._on_confirm_delete)
            return

        name = str(result.get("name", "")).strip()
        if not name:
            return
        replaced = False
        for i, env in enumerate(self.environments):
            if str(env.get("name", "")) == name:
                self.environments[i] = result
                replaced = True
                break
        if not replaced:
            self.environments.append(result)
        self.environments.sort(
            key=lambda env: (0 if str(env.get("name", "")) == DEFAULT_LOCAL_ENVIRONMENT_NAME else 1, str(env.get("name", "")))
        )
        self.selected = name
        self._render_options()

    def _on_confirm_delete(self, confirmed: bool | None) -> None:
        if not confirmed:
            self._pending_delete_name = None
            return
        name = self._pending_delete_name or ""
        if not name or self._is_builtin_local_name(name):
            return
        self.environments = [
            env for env in self.environments if str(env.get("name", "")) != name
        ]
        if self.selected == name:
            self.selected = ""
        self._pending_delete_name = None
        self._render_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        env_options = self.query_one("#env_options", OptionList)
        if event.option_list is env_options:
            self.action_select_current()


class EnvironmentTypeScreen(ModalScreen[str | None]):
    CSS = """
    EnvironmentTypeScreen {
        align: center middle;
    }

    #dialog {
        width: 40;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #types > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #types > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, current_type: str):
        super().__init__()
        self.current_type = current_type

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Select Type")
            yield OptionList(
                Option("ssh", id="type:ssh"),
                Option("vastai", id="type:vastai"),
                id="types",
            )
            yield Static("Enter=select, Esc=cancel", id="hints")

    def on_mount(self) -> None:
        items = self.query_one("#types", OptionList)
        items.highlighted = 1 if self.current_type == "vastai" else 0
        items.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        items = self.query_one("#types", OptionList)
        if event.option_list is not items:
            return
        if event.option_id == "type:vastai":
            self.dismiss("vastai")
        else:
            self.dismiss("ssh")


class EnvironmentEditScreen(ModalScreen[dict[str, Any] | None]):
    CSS = """
    EnvironmentEditScreen {
        align: center middle;
    }

    #editor {
        width: 72;
        height: 78%;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #editor_items {
        width: 1fr;
        height: 1fr;
    }

    #editor_status {
        height: auto;
        color: cyan;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    #editor_items > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #editor_items > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #editor_items.start-focus > .option-list--option-highlighted {
        background: #166534;
        color: #ffffff;
        text-style: bold;
    }

    #editor_items.danger-focus > .option-list--option-highlighted {
        background: #b42318;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, env: dict[str, Any], is_new: bool, default_offer_query: str):
        super().__init__()
        self.env = dict(env)
        self.is_new = is_new
        self.default_offer_query = default_offer_query

    def _is_builtin_local(self) -> bool:
        return str(self.env.get("name", "")).strip() == DEFAULT_LOCAL_ENVIRONMENT_NAME or str(
            self.env.get("type", "")
        ).lower() == "local"

    def compose(self) -> ComposeResult:
        with Vertical(id="editor"):
            yield Static("Environment Editor: Enter to edit/select")
            yield OptionList(id="editor_items")
            yield Static("Enter=select/apply, Esc=back", id="hints")
            yield Static("", id="editor_status")

    def on_mount(self) -> None:
        self._render_items()
        self.query_one("#editor_items", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _set_status(self, text: str) -> None:
        self.query_one("#editor_status", Static).update(text)

    def _short(self, value: str, max_len: int = 40) -> str:
        return value if len(value) <= max_len else value[: max_len - 3] + "..."

    def _env_type(self) -> str:
        t = str(self.env.get("type", "ssh")).lower()
        if t == "local":
            return "local"
        return "vastai" if t == "vastai" else "ssh"

    def _render_items(self) -> None:
        items = self.query_one("#editor_items", OptionList)
        items.clear_options()
        env_type = self._env_type()

        items.add_option(
            Option(_field_label("Name", str(self.env.get("name", ""))), id="field:name")
        )
        if not self._is_builtin_local():
            items.add_option(Option(_field_label("Type", env_type), id="field:type"))
        if env_type == "local":
            items.add_option(Option(_field_label("Mode", "Built-in local environment"), id="field:local_info"))
        elif env_type == "ssh":
            items.add_option(
                Option(
                    _field_label("SSH Host", str(self.env.get("host", ""))),
                    id="field:host",
                )
            )
            items.add_option(
                Option(
                    _field_label("SSH User", str(self.env.get("user", "ubuntu"))),
                    id="field:user",
                )
            )
            items.add_option(
                Option(
                    _field_label("SSH Port", str(self.env.get("port", 22))),
                    id="field:port",
                )
            )
            items.add_option(
                Option(
                    _field_label(
                        "Identity File", str(self.env.get("identity_file", ""))
                    ),
                    id="field:identity_file",
                )
            )
        else:
            items.add_option(
                Option(
                    _field_label(
                        "VastAI Api Key", "***" if self.env.get("api_key") else ""
                    ),
                    id="field:api_key",
                )
            )
            items.add_option(
                Option(
                    _field_label(
                        "VastAI Offer Query",
                        self._short(
                            str(self.env.get("offer_query", self.default_offer_query))
                        ),
                    ),
                    id="field:offer_query",
                )
            )
            items.add_option(
                Option(
                    _field_label("VastAI Order", str(self.env.get("order", "dph"))),
                    id="field:order",
                )
            )
            items.add_option(
                Option(
                    _field_label("VastAI Disk", str(self.env.get("disk", 50))),
                    id="field:disk",
                )
            )
        items.add_option(Option(" ", id="divider:actions", disabled=True))
        items.add_option(Option(_start_label("Save"), id="action:save"))
        if not self._is_builtin_local():
            items.add_option(Option(_danger_label("Delete"), id="danger:delete"))
        items.add_option(Option(_action_label("Back"), id="action:back"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        items = self.query_one("#editor_items", OptionList)
        if event.option_list is not items:
            return
        key = event.option_id
        if key is None:
            return
        self._activate(key)

    def _activate(self, key: str) -> None:
        if key == "action:save":
            name = str(self.env.get("name", "")).strip()
            if not name:
                self._set_status("Name is required.")
                return
            self.dismiss(self._normalized_env())
            return
        if key == "danger:delete":
            self.dismiss(
                {"__delete__": True, "name": str(self.env.get("name", "")).strip()}
            )
            return
        if key == "action:back":
            self.dismiss(None)
            return
        if key == "field:local_info":
            return
        if key == "field:type":
            self.app.push_screen(
                EnvironmentTypeScreen(self._env_type()), self._on_type_selected
            )
            return
        self._edit_field(key.removeprefix("field:"))

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        items = self.query_one("#editor_items", OptionList)
        if event.option_list is not items:
            return
        if event.option_id == "action:save":
            items.add_class("start-focus")
        else:
            items.remove_class("start-focus")
        if event.option_id == "danger:delete":
            items.add_class("danger-focus")
        else:
            items.remove_class("danger-focus")

    def _normalized_env(self) -> dict[str, Any]:
        env_type = self._env_type()
        name = str(self.env.get("name", "")).strip()
        if env_type == "local":
            return {
                "name": DEFAULT_LOCAL_ENVIRONMENT_NAME,
                "type": "local",
            }
        if env_type == "ssh":
            return {
                "name": name,
                "type": "ssh",
                "host": str(self.env.get("host", "")).strip(),
                "user": str(self.env.get("user", "ubuntu")).strip(),
                "port": int(self.env.get("port", 22)),
                "identity_file": str(self.env.get("identity_file", "")).strip(),
            }
        return {
            "name": name,
            "type": "vastai",
            "api_key": str(self.env.get("api_key", "")).strip(),
            "offer_query": str(self.env.get("offer_query", "")).strip()
            or self.default_offer_query,
            "order": str(self.env.get("order", "")).strip() or "dph",
            "disk": int(self.env.get("disk", 50)),
        }

    def _edit_field(self, key: str) -> None:
        title_map = {
            "name": "Edit Name",
            "host": "Edit SSH Host",
            "user": "Edit SSH User",
            "port": "Edit SSH Port",
            "identity_file": "Edit Identity File",
            "api_key": "Edit VastAI Api Key",
            "offer_query": "Edit VastAI Offer Query",
            "order": "Edit VastAI Order",
            "disk": "Edit VastAI Disk",
        }
        value = str(self.env.get(key, ""))
        if self._is_builtin_local():
            return
        if key == "identity_file":
            self.app.push_screen(
                PathEditScreen(title_map.get(key, "Edit"), value),
                lambda result, k=key: self._on_field_edited(k, result),
            )
            return
        self.app.push_screen(
            EditValueScreen(
                title_map.get(key, "Edit"),
                value,
                password=(key == "api_key"),
            ),
            lambda result, k=key: self._on_field_edited(k, result),
        )

    def _on_field_edited(self, key: str, result: str | None) -> None:
        if result is None:
            return
        if key in {"port", "disk"}:
            try:
                value = int(result)
            except ValueError:
                self._set_status("Port/Disk must be integers.")
                return
            if key == "port" and (value < 1 or value > 65535):
                self._set_status("Port values must be in range 1..65535.")
                return
            self.env[key] = value
        else:
            self.env[key] = result
        self._render_items()

    def _on_type_selected(self, selected: str | None) -> None:
        if selected is None:
            return
        self.env["type"] = selected
        if selected == "ssh":
            self.env.setdefault("host", "")
            self.env.setdefault("user", "ubuntu")
            self.env.setdefault("port", 22)
            self.env.setdefault("identity_file", "")
        else:
            self.env.setdefault("api_key", "")
            self.env.setdefault("offer_query", self.default_offer_query)
            self.env.setdefault("order", "dph")
            self.env.setdefault("disk", 50)
        self._render_items()


class OtherOptionsScreen(ModalScreen[dict[str, str] | None]):
    CSS = """
    OtherOptionsScreen {
        align: center middle;
    }

    #dialog {
        width: 72;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #items > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #items > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, options: dict[str, str]):
        super().__init__()
        self.options = {
            "hf_token": str(options.get("hf_token", "")),
            "civitai_api_key": str(options.get("civitai_api_key", "")),
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Other Options")
            yield OptionList(id="items")
            yield Static("Enter=edit/save, Esc=back", id="hints")

    def on_mount(self) -> None:
        self._render_items()
        self.query_one("#items", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _render_items(self) -> None:
        items = self.query_one("#items", OptionList)
        items.clear_options()
        items.add_option(
            Option(
                _field_label(
                    "HF Token",
                    "***" if self.options.get("hf_token", "").strip() else "",
                ),
                id="field:hf_token",
            )
        )
        items.add_option(
            Option(
                _field_label(
                    "CivitAI API Key",
                    "***" if self.options.get("civitai_api_key", "").strip() else "",
                ),
                id="field:civitai_api_key",
            )
        )
        items.add_option(Option(" ", id="divider:actions", disabled=True))
        items.add_option(Option(_start_label("Save"), id="action:save"))
        items.add_option(Option(_action_label("Back"), id="action:back"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        items = self.query_one("#items", OptionList)
        if event.option_list is not items:
            return
        key = event.option_id
        if key is None:
            return
        if key == "action:save":
            self.dismiss(self.options)
            return
        if key == "action:back":
            self.dismiss(None)
            return
        field = key.removeprefix("field:")
        self.app.push_screen(
            EditValueScreen(
                "Edit HF Token" if field == "hf_token" else "Edit CivitAI API Key",
                self.options.get(field, ""),
                password=True,
            ),
            lambda result, f=field: self._on_field_edited(f, result),
        )

    def _on_field_edited(self, field: str, result: str | None) -> None:
        if result is None:
            return
        self.options[field] = result
        self._render_items()


class TrainLauncherApp(App[LauncherResult]):
    CSS = """
    Screen {
        padding: 1;
    }

    #root {
        height: 1fr;
    }

    #menu {
        height: 1fr;
        border: solid #666666;
        padding: 1;
    }

    #menu_title {
        margin-bottom: 1;
    }

    #items {
        height: 1fr;
    }

    #status {
        height: auto;
        color: cyan;
        margin-top: 0;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    #items > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #items > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #items > .option-list--option-disabled {
        color: #ffd166;
        text-style: bold;
    }

    #items.start-focus > .option-list--option-highlighted {
        background: #166534;
        color: #ffffff;
        text-style: bold;
    }

    #items.danger-focus > .option-list--option-highlighted {
        background: #b42318;
        color: #ffffff;
        text-style: bold;
    }

    """

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit", show=False),
        Binding("ctrl+q", "request_quit", "Quit", show=False),
        Binding("enter", "activate_selected", "Select"),
    ]

    ITEM_ENV = "env"
    ITEM_TRAIN = "train"
    ITEM_SCRIPT = "script"
    ITEM_OPTIONS = "options"
    ITEM_TAGGER = "tagger"
    ITEM_START = "start"
    ITEM_QUIT = "quit"

    def __init__(
        self,
        *,
        environments: list[dict[str, Any]],
        last: dict[str, str],
        other_options: dict[str, str],
        default_offer_query: str,
        train_script_options: list[str],
    ):
        super().__init__()
        self.environments = [dict(env) for env in environments]
        self.last = dict(last)
        self.other_options = {
            "hf_token": str(other_options.get("hf_token", "")),
            "civitai_api_key": str(other_options.get("civitai_api_key", "")),
        }
        self.default_offer_query = default_offer_query
        self.train_script_options = train_script_options

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="menu"):
                yield OptionList(id="items")
                yield Static("Enter=select/apply, Esc=quit", id="hints")
            yield Static("", id="status")

    def on_mount(self) -> None:
        self._render_items()
        self.query_one("#items", OptionList).focus()
        self._set_status("")

    def _env_names(self) -> list[str]:
        return [
            str(env.get("name", "")) for env in self.environments if env.get("name")
        ]

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _render_items(self) -> None:
        option_list = self.query_one("#items", OptionList)
        option_list.clear_options()

        environment_name = self.last.get("environment_name", "")
        train_config = self.last.get("train_config_path", "") or "train.toml"
        train_script = self.last.get("train_script", "")
        if not train_script and self.train_script_options:
            preferred = "train_network.py"
            train_script = (
                preferred
                if preferred in self.train_script_options
                else self.train_script_options[0]
            )
            self.last["train_script"] = train_script

        option_list.add_option(
            Option(
                _field_label("Environment", environment_name or "(not selected)"),
                id="field:env",
            )
        )
        option_list.add_option(
            Option(_field_label("Train Config", train_config), id="field:train_config")
        )
        option_list.add_option(
            Option(
                _field_label("Train Script", train_script or "(not selected)"),
                id="field:train_script",
            )
        )
        option_list.add_option(
            Option(
                _field_label(
                    "Other Options",
                    f"hf={'on' if self.other_options.get('hf_token', '').strip() else 'off'}, "
                    f"civitai={'on' if self.other_options.get('civitai_api_key', '').strip() else 'off'}",
                ),
                id="field:other_options",
            )
        )
        option_list.add_option(Option(" ", id="divider:actions", disabled=True))
        option_list.add_option(
            Option(_action_label("Tagger Workspace"), id="action:tagger")
        )
        option_list.add_option(Option(_start_label("Start Train"), id="action:start"))
        option_list.add_option(Option(_danger_label("Quit"), id="danger:quit"))

    def _item_from_option_id(self, option_id: str | None) -> str | None:
        mapping = {
            "field:env": self.ITEM_ENV,
            "field:train_config": self.ITEM_TRAIN,
            "field:train_script": self.ITEM_SCRIPT,
            "field:other_options": self.ITEM_OPTIONS,
            "action:tagger": self.ITEM_TAGGER,
            "action:start": self.ITEM_START,
            "danger:quit": self.ITEM_QUIT,
        }
        if option_id is None:
            return None
        return mapping.get(option_id)

    def action_activate_selected(self) -> None:
        focused = self.focused
        if not isinstance(focused, OptionList):
            return
        index = focused.highlighted
        if index is None:
            return
        option = focused.get_option_at_index(index)
        item = self._item_from_option_id(option.id)
        if item is None:
            return
        self._activate_item(item)

    def _edit_item(self, item: str) -> None:
        if item == self.ITEM_ENV:
            self._select_environment()
            return

        if item == self.ITEM_TRAIN:
            value = self.last.get("train_config_path", "") or "train.toml"
            self.push_screen(
                PathEditScreen("Edit Train Config Path", value),
                self._on_train_config_edited,
            )
            return

        if item == self.ITEM_SCRIPT:
            value = self.last.get("train_script", "")
            if not self.train_script_options:
                self._set_status("No train scripts are available from GitHub scan.")
                return
            self.push_screen(
                SelectValueScreen(
                    "Select Train Script",
                    options=self.train_script_options,
                    selected=value,
                ),
                self._on_train_script_selected,
            )
            return

        if item == self.ITEM_OPTIONS:
            self.push_screen(
                OtherOptionsScreen(self.other_options),
                self._on_other_options_saved,
            )
            return

        self._set_status("This item is not editable.")

    def _activate_item(self, item: str) -> None:
        if item == self.ITEM_START:
            self._start_train()
            return

        if item == self.ITEM_TAGGER:
            self._open_tagger_workspace()
            return

        if item == self.ITEM_QUIT:
            self.action_request_quit()
            return

        self._edit_item(item)

    def _select_environment(self) -> None:
        selected = self.last.get("environment_name", "")
        self.push_screen(
            SelectEnvironmentScreen(
                environments=self.environments,
                selected=selected,
                default_offer_query=self.default_offer_query,
            ),
            self._on_environment_selected,
        )

    def _on_train_config_edited(self, edited: str | None) -> None:
        if edited is None:
            return
        self.last["train_config_path"] = edited
        self._render_items()

    def _on_train_script_selected(self, edited: str | None) -> None:
        if edited is None:
            return
        self.last["train_script"] = edited
        self._render_items()

    def _on_environment_selected(self, result: EnvironmentPickerResult | None) -> None:
        if result is None:
            return
        self.environments = result.environments
        if result.selected_name is None:
            self._render_items()
            return
        self.last["environment_name"] = result.selected_name
        self._render_items()

    def _start_train(self) -> None:
        env_name = self.last.get("environment_name", "")
        if not env_name:
            self._set_status("Environment is required.")
            return

        if not any(name == env_name for name in self._env_names()):
            self._set_status("Selected environment does not exist.")
            return

        script = self.last.get("train_script", "")
        if not script:
            self._set_status("Train script is required.")
            return
        if self.train_script_options and script not in self.train_script_options:
            self._set_status("Selected train script is not in GitHub script list.")
            return

        selection = RunSelection(
            environment_name=env_name,
            train_config_path=self.last.get("train_config_path", "") or "train.toml",
            train_script=script,
        )

        self.exit(
            LauncherResult(
                action="start",
                environments=self.environments,
                last=self.last,
                other_options=self.other_options,
                selection=selection,
            )
        )

    def _open_tagger_workspace(self) -> None:
        self.exit(
            LauncherResult(
                action="tagger",
                environments=self.environments,
                last=self.last,
                other_options=self.other_options,
                selection=None,
            )
        )

    def action_request_quit(self) -> None:
        self.exit(
            LauncherResult(
                action="quit",
                environments=self.environments,
                last=self.last,
                other_options=self.other_options,
                selection=None,
            )
        )

    def _on_other_options_saved(self, options: dict[str, str] | None) -> None:
        if options is None:
            return
        self.other_options = {
            "hf_token": str(options.get("hf_token", "")).strip(),
            "civitai_api_key": str(options.get("civitai_api_key", "")).strip(),
        }
        self._render_items()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        items = self.query_one("#items", OptionList)
        if event.option_list is not items:
            return
        item = self._item_from_option_id(event.option_id)
        if item is None:
            return
        self._activate_item(item)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        items = self.query_one("#items", OptionList)
        if event.option_list is not items:
            return
        if event.option_id == "action:start":
            items.add_class("start-focus")
        else:
            items.remove_class("start-focus")
        if event.option_id == "danger:quit":
            items.add_class("danger-focus")
        else:
            items.remove_class("danger-focus")


class ConfirmActionScreen(ModalScreen[bool]):
    CSS = """
    ConfirmActionScreen {
        align: center middle;
    }

    #dialog {
        width: 72;
        height: auto;
        border: solid #666666;
        background: $surface;
        padding: 1;
    }

    #options {
        margin-top: 1;
    }

    #options > .option-list--option {
        padding: 0 1;
        color: #d8d8d8;
    }

    #options > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.message)
            yield OptionList(
                Option(_start_label("Proceed"), id="action:yes"),
                Option(_action_label("Cancel"), id="action:no"),
                id="options",
            )

    def on_mount(self) -> None:
        self.query_one("#options", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id == "action:yes":
            self.dismiss(True)
            return
        self.dismiss(False)


class TaggerWorkspaceApp(App[TaggerWorkspaceResult]):
    CSS = """
    Screen {
        padding: 1;
    }

    #root {
        height: 1fr;
    }

    #toolbar {
        height: auto;
        border: solid #666666;
        padding: 0 1;
        margin-bottom: 1;
    }

    #body {
        height: 1fr;
    }

    #actions_panel {
        width: 38;
        border: solid #666666;
        padding: 1;
        margin-right: 1;
    }

    #actions {
        height: 1fr;
    }

    #stats_panel {
        width: 1fr;
        border: solid #666666;
        padding: 1;
    }

    #stats_text {
        height: auto;
        color: #c8d6e5;
    }

    #tag_list {
        height: 1fr;
        margin-top: 1;
    }

    #result {
        height: auto;
        color: cyan;
        margin-top: 1;
    }

    #hints {
        height: auto;
        color: #999999;
        margin-top: 1;
    }

    #actions > .option-list--option-highlighted {
        background: #0d6efd;
        color: #ffffff;
        text-style: bold;
    }

    #tag_list > .option-list--option-highlighted {
        background: #164e63;
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit_workspace", "Quit", show=False),
        Binding("ctrl+q", "quit_workspace", "Quit", show=False),
        Binding("escape", "leave_workspace", "Back", show=False),
        Binding("E", "rename_selected_tag", "Rename Tag", show=False),
        Binding("X", "delete_selected_tag", "Delete Tag", show=False),
    ]

    def __init__(
        self,
        *,
        dataset_dir: str,
        model: str,
        threshold: float,
        batch: int,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.model = model
        self.threshold = threshold
        self.batch = batch
        self.stats = TagStatsSnapshot(total_images=0, captioned_images=0, tags=[])
        self._tag_ids: list[str] = []
        self._selected_tag: str | None = None
        self._auto_tag_running = False

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield Static("", id="toolbar")
            with Horizontal(id="body"):
                with Vertical(id="actions_panel"):
                    yield Static("Actions")
                    yield OptionList(id="actions")
                    yield ProgressBar(total=1, id="progress")
                with Vertical(id="stats_panel"):
                    yield Static("", id="stats_text")
                    yield OptionList(id="tag_list")
            yield Static(
                "Enter=select/apply, Shift+E=rename tag, Shift+X=delete tag, Esc=back, Ctrl+Q=quit",
                id="hints",
            )
            yield Static("", id="result")

    def on_mount(self) -> None:
        self._render_toolbar()
        self._render_actions()
        self._refresh_stats()
        self._set_progress(0, 1)
        self.query_one("#actions", OptionList).focus()

    def _set_result(self, text: str) -> None:
        self.query_one("#result", Static).update(text)

    def _set_progress(self, done: int, total: int) -> None:
        bar = self.query_one("#progress", ProgressBar)
        safe_total = max(1, total)
        safe_done = max(0, min(done, safe_total))
        bar.update(total=safe_total, progress=safe_done)

    def _render_toolbar(self) -> None:
        path = self.dataset_dir or "(not selected)"
        self.query_one("#toolbar", Static).update(
            f"Dataset: {path} | Model: {self.model} | threshold={self.threshold:.2f} | batch={self.batch}"
        )

    def _render_actions(self) -> None:
        actions = self.query_one("#actions", OptionList)
        actions.clear_options()
        actions.add_option(Option("Set Dataset Path", id="field:dataset"))
        actions.add_option(Option("Set Tagger Model", id="field:model"))
        actions.add_option(Option("Set Threshold", id="field:threshold"))
        actions.add_option(Option("Set Batch Size", id="field:batch"))
        actions.add_option(Option(" ", id="divider:run", disabled=True))
        disable_during_run = self._auto_tag_running
        actions.add_option(
            Option(
                _start_label("Auto Tag"),
                id="run:tag",
                disabled=disable_during_run,
            )
        )
        actions.add_option(Option("Add Tags", id="run:add", disabled=disable_during_run))
        actions.add_option(
            Option("Move Tags to Front", id="run:front", disabled=disable_during_run)
        )
        actions.add_option(
            Option("Shuffle Tags", id="run:shuffle", disabled=disable_during_run)
        )
        actions.add_option(
            Option("Refresh Stats", id="run:refresh", disabled=disable_during_run)
        )
        actions.add_option(Option(" ", id="divider:delete", disabled=True))
        can_delete = self._selected_tag is not None and not disable_during_run
        can_rename = can_delete
        actions.add_option(
            Option(
                _action_label("Rename Selected Tag")
                if can_rename
                else "Rename Selected Tag (select from list)",
                id="run:rename",
                disabled=not can_rename,
            )
        )
        actions.add_option(
            Option(
                _danger_label("Delete Selected Tag") if can_delete else "Delete Selected Tag (select from list)",
                id="run:delete",
                disabled=not can_delete,
            )
        )
        actions.add_option(
            Option(
                _danger_label("Delete All Tags"),
                id="run:delete_all",
                disabled=disable_during_run,
            )
        )
        actions.add_option(Option(" ", id="divider:nav", disabled=True))
        actions.add_option(Option(_action_label("Back"), id="nav:back"))
        actions.add_option(Option(_danger_label("Quit"), id="nav:quit"))

    def _refresh_stats(self) -> None:
        if not self.dataset_dir:
            self.stats = TagStatsSnapshot(total_images=0, captioned_images=0, tags=[])
            self._tag_ids = []
            self._selected_tag = None
            self.query_one("#stats_text", Static).update("Dataset is not selected.")
            self.query_one("#tag_list", OptionList).clear_options()
            self._render_actions()
            return

        self.stats = collect_stats(self.dataset_dir)
        ratio = 0.0
        if self.stats.total_images > 0:
            ratio = (self.stats.captioned_images / self.stats.total_images) * 100.0
        self.query_one("#stats_text", Static).update(
            "Tag Stats\n"
            f"- Images: {self.stats.total_images}\n"
            f"- Captioned: {self.stats.captioned_images} ({ratio:.1f}%)\n"
            f"- Unique tags: {len(self.stats.tags)}"
        )
        self._render_tag_list()
        self._render_actions()

    def _render_tag_list(self) -> None:
        tag_list = self.query_one("#tag_list", OptionList)
        tag_list.clear_options()
        self._tag_ids = []
        selected_exists = False
        for idx, (tag, count) in enumerate(self.stats.tags):
            tag_list.add_option(Option(f"{tag}  ({count})", id=f"tag:{idx}"))
            self._tag_ids.append(tag)
            if self._selected_tag == tag:
                tag_list.highlighted = idx
                selected_exists = True

        if not self._tag_ids:
            self._selected_tag = None
            tag_list.add_option(Option("(no tags)", id="tag:none", disabled=True))
        elif not selected_exists:
            tag_list.highlighted = 0
            self._selected_tag = self._tag_ids[0]

    def _ensure_dataset(self) -> bool:
        if not self.dataset_dir:
            self._set_result("Dataset path is required.")
            return False
        path = Path(self.dataset_dir).expanduser()
        if not path.is_dir():
            self._set_result(f"Dataset directory not found: {path}")
            return False
        self.dataset_dir = str(path.resolve())
        self._render_toolbar()
        return True

    def _print_summary(self, summary: TaggerRunSummary) -> None:
        message = (
            f"{summary.command}: processed={summary.processed_images}, "
            f"changed={summary.changed_captions}, failed={summary.failed_images}, "
            f"duration={summary.duration_seconds:.1f}s"
        )
        if summary.message:
            message += f" | {summary.message}"
        if summary.failed_paths:
            failed = ", ".join(summary.failed_paths[:3])
            if len(summary.failed_paths) > 3:
                failed += ", ..."
            message += f" | failed samples: {failed}"
        self._set_result(message)

    def _run_auto_tag(self) -> None:
        if not self._ensure_dataset():
            return
        overwrite = count_overwrite_candidates(self.dataset_dir)
        self.push_screen(
            ConfirmActionScreen(
                f"Auto Tag will append tags. {overwrite} images already have captions. Proceed?"
            ),
            self._on_confirm_auto_tag,
        )

    def _on_confirm_auto_tag(self, confirmed: bool | None) -> None:
        if not confirmed:
            self._set_result("Auto Tag canceled.")
            return
        self._auto_tag_running = True
        self._set_progress(0, max(1, self.stats.total_images))
        self._set_result("Auto Tag started...")
        self._render_actions()
        asyncio.create_task(self._run_auto_tag_async())

    async def _run_auto_tag_async(self) -> None:
        def progress(done: int, total: int, current: str) -> None:
            self.call_from_thread(self._on_auto_tag_progress, done, total, current)

        try:
            summary = await asyncio.to_thread(
                auto_tag,
                self.dataset_dir,
                TaggerModelConfig(
                    model=self.model,
                    threshold=self.threshold,
                    batch=self.batch,
                ),
                progress,
            )
            self._print_summary(summary)
            self._refresh_stats()
        except Exception as exc:
            self._set_result(f"Auto Tag failed: {exc}")
        finally:
            self._auto_tag_running = False
            self._render_actions()

    def _on_auto_tag_progress(self, done: int, total: int, current: str) -> None:
        self._set_progress(done, total)
        if current:
            self._set_result(f"Auto Tag: {done}/{total} | {Path(current).name}")
        else:
            self._set_result(f"Auto Tag: {done}/{total}")

    def _run_with_tag_input(self, run_type: str, title: str) -> None:
        if not self._ensure_dataset():
            return
        self.push_screen(
            EditValueScreen(title, ""),
            lambda result, command=run_type: self._on_tags_entered(command, result),
        )

    def _on_tags_entered(self, command: str, entered: str | None) -> None:
        if entered is None:
            return
        tags = parse_tag_input(entered)
        if not tags:
            self._set_result("At least one tag is required.")
            return
        if command == "add":
            summary = add_tags(self.dataset_dir, tags)
        else:
            summary = front_tags(self.dataset_dir, tags)
        self._print_summary(summary)
        self._refresh_stats()

    def _run_shuffle(self) -> None:
        if not self._ensure_dataset():
            return
        summary = shuffle_tags(self.dataset_dir)
        self._print_summary(summary)
        self._refresh_stats()

    def _run_delete_selected(self) -> None:
        if not self._ensure_dataset() or not self._selected_tag:
            self._set_result("Select a tag from the list first.")
            return
        selected = self._selected_tag
        self.push_screen(
            ConfirmActionScreen(
                f"Delete selected tag '{selected}' from all captions?"
            ),
            lambda confirmed, tag=selected: self._on_confirm_delete_tag(tag, confirmed),
        )

    def _on_confirm_delete_tag(self, tag: str, confirmed: bool | None) -> None:
        if not confirmed:
            self._set_result("Delete canceled.")
            return
        summary = remove_single_tag(self.dataset_dir, tag)
        self._print_summary(summary)
        self._refresh_stats()

    def _run_rename_selected(self) -> None:
        if not self._ensure_dataset() or not self._selected_tag:
            self._set_result("Select a tag from the list first.")
            return
        selected = self._selected_tag
        self.push_screen(
            EditValueScreen("Rename Selected Tag", selected),
            lambda value, old=selected: self._on_rename_selected(old, value),
        )

    def _on_rename_selected(self, old_tag: str, value: str | None) -> None:
        if value is None:
            return
        new_tag = value.strip()
        if not new_tag:
            self._set_result("Tag name cannot be empty.")
            return
        summary = rename_single_tag(self.dataset_dir, old_tag, new_tag)
        self._print_summary(summary)
        self._selected_tag = new_tag
        self._refresh_stats()

    def _run_delete_all_tags(self) -> None:
        if not self._ensure_dataset():
            return
        self.push_screen(
            ConfirmActionScreen("Delete all tags from all captions in this dataset?"),
            self._on_confirm_delete_all_tags,
        )

    def _on_confirm_delete_all_tags(self, confirmed: bool | None) -> None:
        if not confirmed:
            self._set_result("Delete all canceled.")
            return
        summary = delete_all_tags(self.dataset_dir)
        self._print_summary(summary)
        self._refresh_stats()

    def _edit_dataset(self) -> None:
        current = self.dataset_dir or "."
        self.push_screen(
            PathEditScreen("Set Dataset Directory", current, directories_only=True),
            self._on_dataset_selected,
        )

    def _on_dataset_selected(self, value: str | None) -> None:
        if value is None:
            return
        path = Path(value).expanduser()
        self.dataset_dir = str(path.resolve()) if path.exists() else str(path)
        self._selected_tag = None
        self._render_toolbar()
        self._refresh_stats()

    def _edit_model(self) -> None:
        self.push_screen(
            EditValueScreen("Set Tagger Model", self.model),
            self._on_model_edited,
        )

    def _on_model_edited(self, value: str | None) -> None:
        if value is None:
            return
        model = value.strip()
        if not model:
            self._set_result("Model cannot be empty.")
            return
        self.model = model
        self._render_toolbar()

    def _edit_threshold(self) -> None:
        self.push_screen(
            EditValueScreen("Set Threshold (0.0 - 1.0)", str(self.threshold)),
            self._on_threshold_edited,
        )

    def _on_threshold_edited(self, value: str | None) -> None:
        if value is None:
            return
        try:
            threshold = float(value)
        except ValueError:
            self._set_result("Threshold must be a float.")
            return
        if threshold < 0.0 or threshold > 1.0:
            self._set_result("Threshold must be in range 0.0..1.0.")
            return
        self.threshold = threshold
        self._render_toolbar()

    def _edit_batch(self) -> None:
        self.push_screen(
            EditValueScreen("Set Batch Size", str(self.batch)),
            self._on_batch_edited,
        )

    def _on_batch_edited(self, value: str | None) -> None:
        if value is None:
            return
        try:
            batch = int(value)
        except ValueError:
            self._set_result("Batch must be an integer.")
            return
        if batch < 1:
            self._set_result("Batch must be >= 1.")
            return
        self.batch = batch
        self._render_toolbar()

    def action_leave_workspace(self) -> None:
        self.exit(
            TaggerWorkspaceResult(
                action="back",
                dataset_dir=self.dataset_dir,
                model=self.model,
                threshold=self.threshold,
                batch=self.batch,
            )
        )

    def action_quit_workspace(self) -> None:
        self.exit(
            TaggerWorkspaceResult(
                action="quit",
                dataset_dir=self.dataset_dir,
                model=self.model,
                threshold=self.threshold,
                batch=self.batch,
            )
        )

    def action_delete_selected_tag(self) -> None:
        if self._auto_tag_running:
            self._set_result("Auto Tag is running. Wait until it finishes.")
            return
        self._run_delete_selected()

    def action_rename_selected_tag(self) -> None:
        if self._auto_tag_running:
            self._set_result("Auto Tag is running. Wait until it finishes.")
            return
        self._run_rename_selected()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        actions = self.query_one("#actions", OptionList)
        tag_list = self.query_one("#tag_list", OptionList)
        if event.option_list is actions:
            option_id = event.option_id or ""
            if option_id == "field:dataset":
                self._edit_dataset()
            elif option_id == "field:model":
                self._edit_model()
            elif option_id == "field:threshold":
                self._edit_threshold()
            elif option_id == "field:batch":
                self._edit_batch()
            elif option_id == "run:tag":
                self._run_auto_tag()
            elif option_id == "run:add":
                self._run_with_tag_input("add", "Add Tags (comma-separated)")
            elif option_id == "run:front":
                self._run_with_tag_input("front", "Move Tags to Front (comma-separated)")
            elif option_id == "run:shuffle":
                self._run_shuffle()
            elif option_id == "run:refresh":
                self._refresh_stats()
                self._set_result("Stats refreshed.")
            elif option_id == "run:delete":
                self._run_delete_selected()
            elif option_id == "run:rename":
                self._run_rename_selected()
            elif option_id == "run:delete_all":
                self._run_delete_all_tags()
            elif option_id == "nav:back":
                self.action_leave_workspace()
            elif option_id == "nav:quit":
                self.action_quit_workspace()
            return

        if event.option_list is tag_list:
            option_id = event.option_id or ""
            if not option_id.startswith("tag:"):
                return
            try:
                index = int(option_id.removeprefix("tag:"))
            except ValueError:
                return
            if index < 0 or index >= len(self._tag_ids):
                return
            self._selected_tag = self._tag_ids[index]
            self._render_actions()
            self._set_result(f"Selected tag: {self._selected_tag}")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        tag_list = self.query_one("#tag_list", OptionList)
        if event.option_list is not tag_list:
            return
        option_id = event.option_id or ""
        if not option_id.startswith("tag:"):
            return
        try:
            index = int(option_id.removeprefix("tag:"))
        except ValueError:
            return
        if index < 0 or index >= len(self._tag_ids):
            return
        self._selected_tag = self._tag_ids[index]
        self._render_actions()
        self._set_result(f"Selected tag: {self._selected_tag}")
