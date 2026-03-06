from __future__ import annotations

import ast
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "sd_train"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / PACKAGE
DOT_OUTPUT = ROOT / "docs" / "dependency-graph.dot"
PNG_OUTPUT = ROOT / "docs" / "dependency-graph.png"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    stem = rel.with_suffix("")
    parts = list(stem.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parse_edges(path: Path) -> set[Edge]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    src = _module_name(path)
    edges: set[Edge] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(PACKAGE):
                    edges.add(Edge(src=src, dst=name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(PACKAGE):
                edges.add(Edge(src=src, dst=node.module))

    return edges


def _node_id(module_name: str) -> str:
    return "N_" + module_name.replace(".", "_").replace("-", "_")


def _build_dot(edges: list[Edge]) -> str:
    nodes = sorted({e.src for e in edges} | {e.dst for e in edges})

    lines = [
        "digraph DependencyGraph {",
        "  rankdir=LR;",
        '  graph [fontsize=10, fontname="Helvetica"];',
        '  node [shape=box, style="rounded", fontsize=10, fontname="Helvetica"];',
        '  edge [fontsize=9, fontname="Helvetica"];',
        "",
    ]

    for node in nodes:
        nid = _node_id(node)
        label = node.replace('"', r'\"')
        lines.append(f'  {nid} [label="{label}"];')

    lines.append("")

    for edge in edges:
        lines.append(f"  {_node_id(edge.src)} -> {_node_id(edge.dst)};")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _generate_dot() -> None:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    edges: set[Edge] = set()
    for file in files:
        edges |= _parse_edges(file)

    normalized = sorted(edges, key=lambda e: (e.src, e.dst))
    DOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOT_OUTPUT.write_text(_build_dot(normalized), encoding="utf-8")
    print(f"Wrote {DOT_OUTPUT}")


def _render_png() -> None:
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise RuntimeError("dot not found in PATH. Install graphviz first.")

    subprocess.run([dot_bin, "-Tpng", str(DOT_OUTPUT), "-o", str(PNG_OUTPUT)], check=True)
    print(f"Wrote {PNG_OUTPUT}")


def main() -> None:
    _generate_dot()
    _render_png()


if __name__ == "__main__":
    main()
