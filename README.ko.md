# sd-train

[English](README.md) | [한국어](README.ko.md)

`sd-train`은 로컬 또는 원격 머신에서 SDXL/LoRA 학습 작업을 실행하기 위한 터미널 UI입니다. 학습 환경을 선택하고, 실행 전 설정을 검증하고, `kohya-ss/sd-scripts` 기반 학습을 시작한 뒤, 결과물을 로컬로 동기화할 수 있습니다.

## 할 수 있는 일

- 로컬 머신, SSH 서버 또는 Vast.ai 인스턴스에서 학습 실행
- 런처에서 `train.toml` 선택
- `kohya-ss/sd-scripts`의 학습 스크립트 선택
- Hugging Face 또는 CivitAI에서 모델/데이터셋 자산 내려받기
- 학습 결과를 로컬 `outputs/<run-id>` 디렉터리로 동기화
- 내장 태거 워크스페이스로 데이터셋 캡션 작업 수행

## 요구 사항

- Python 3.12+
- `uv`
- 로컬 `rclone`
- 원격 환경을 사용할 경우 다음 도구 필요:
  - `python3`
  - `rclone`
  - `git`

학습 시작 시 선택한 머신의 `~/.sd-train/sd-scripts`는 자동으로 준비됩니다. `uv`가 없으면 자동 설치를 시도합니다.

## 설치

```bash
uv sync
```

## 실행

```bash
uv run launch
```

아래 명령으로도 실행할 수 있습니다.

```bash
uv run -m sd_train.cli
```

첫 실행 시 프로젝트 루트에 `config.toml`이 자동 생성됩니다.

## 빠른 시작

1. `uv run launch`를 실행합니다.
2. 내장 `local` 환경을 선택하거나 원격 환경을 만듭니다.
3. `train.toml`을 선택합니다.
4. 학습 스크립트를 선택합니다.
5. Hugging Face 토큰이나 CivitAI API 키가 필요하면 `Other Options`를 엽니다.
6. preflight 검사를 통과하면 학습을 시작합니다.

## Hugging Face / CivitAI 모델 사용법

모델 파일을 미리 직접 내려받지 않아도, `train.toml`에서 원격 자산을 직접 참조할 수 있습니다. `sd-train`은 preflight 단계에서 참조 형식을 검사하고 실제 접근 가능한지도 확인합니다.

### Hugging Face

사용 가능한 형식:

- `org/repo`
- `model:org/repo`
- `dataset:org/repo`
- `model:org/repo@revision`
- `model:org/repo::file.safetensors`
- `model:org/repo@revision::file.safetensors`

예시:

```toml
pretrained_model_name_or_path = "model:stabilityai/stable-diffusion-xl-base-1.0"
network_weights = "model:some-user/my-lora::my-lora.safetensors"
dataset_config = "dataset:some-user/my-dataset"
```

참고:

- `.safetensors` 같은 파일 경로가 필요한 키에는 `::subpath` 형식을 사용하세요.
- 비공개 또는 gated 리포지토리라면 `Other Options`에 Hugging Face 토큰을 입력해야 합니다.
- `org/repo`만 써도 Hugging Face 모델 참조로 처리됩니다.

### CivitAI

사용 가능한 형식:

- `civitai:46846`
- `civitai:https://civitai.com/models/1234567?modelVersionId=46846`
- `civitai:https://civitai.com/api/download/models/46846`
- `civitai:46846::my-model.safetensors`

예시:

```toml
pretrained_model_name_or_path = "civitai:46846"
network_weights = "civitai:46846::my-lora.safetensors"
```

참고:

- CivitAI 다운로드를 쓰려면 `Other Options`에 `civitai_api_key`를 설정해야 합니다.
- 모델 페이지 URL을 넣으면 `sd-train`이 내려받기 가능한 version id로 자동 해석을 시도합니다.
- `::filename`은 선택 사항이지만, 저장 파일명을 고정하고 싶을 때 유용합니다.

## 태거 워크스페이스

런처에서 `Tagger Workspace`를 열면 다음 작업을 할 수 있습니다.

- 데이터셋 태그 통계 확인
- 이미지 자동 태깅
- 캡션 일괄 수정

기본 태거 모델은 `SmilingWolf/wd-vit-tagger-v3`입니다.

## 참고

- 인증 정보는 `config.toml`에 평문으로 저장됩니다.
- 실제 토큰이나 API 키는 커밋하지 않는 편이 좋습니다.
- CivitAI 다운로드에는 `civitai_api_key`가 필요합니다.
