# sd-train

[English](README.md) | [한국어](README.ko.md)

`sd-train`은 원격 머신에서 SDXL/LoRA 학습 작업을 실행하기 위한 터미널 UI입니다. 학습 환경을 선택하고, 실행 전 설정을 검증하고, 원격에서 `kohya-ss/sd-scripts` 기반 학습을 시작한 뒤, 결과물을 로컬로 동기화할 수 있습니다.

## 할 수 있는 일

- SSH 서버 또는 Vast.ai 인스턴스에서 학습 실행
- 런처에서 `train.toml` 선택
- `kohya-ss/sd-scripts`의 학습 스크립트 선택
- Hugging Face 또는 CivitAI에서 모델/데이터셋 자산 내려받기
- 학습 결과를 로컬 `outputs/<run-id>` 디렉터리로 동기화
- 내장 태거 워크스페이스로 데이터셋 캡션 작업 수행

## 요구 사항

- Python 3.12+
- `uv`
- 로컬 `rclone`
- 원격 환경에 다음 도구 필요:
  - `python3`
  - `rclone`
  - `git`

학습 시작 시 원격의 `~/.sd-train/sd-scripts`는 자동으로 준비됩니다. 원격에 `uv`가 없으면 자동 설치를 시도합니다.

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
2. 환경을 만들거나 기존 환경을 선택합니다.
3. `train.toml`을 선택합니다.
4. 학습 스크립트를 선택합니다.
5. Hugging Face 토큰이나 CivitAI API 키가 필요하면 `Other Options`를 엽니다.
6. preflight 검사를 통과하면 학습을 시작합니다.

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
