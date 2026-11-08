# VLM 운용 안내

## 빠른 실행

VS Code 터미널에서 다음을 실행합니다.

```bash
cd /home/work/123-0125
bash run_vlm.sh
```

기본 모델은 `Qwen/Qwen2.5-VL-3B-Instruct`이고, 기본 입력은 같은 폴더의 `test_image.jpg`입니다. 실행이 끝나면 `===== 결과 =====` 아래에 이미지 설명이 표시됩니다.

## 내 이미지 사용

이미지를 이 작업 폴더에 업로드한 뒤 파일 경로를 지정합니다. 공백이 있는 경로는 따옴표로 감쌉니다.

```bash
bash run_vlm.sh --image ./my_photo.jpg --prompt "이 이미지에 무엇이 보이는지 한국어로 설명해줘."
```

외부 이미지 URL도 `--image`로 전달할 수 있지만, 해당 사이트가 접근을 거부하면 실패하므로 로컬 파일을 권장합니다.

## 실행 환경

| 항목 | 값 |
| --- | --- |
| 작업 폴더 | `/home/work/123-0125` |
| 전용 가상환경 | `.venv-vlm` |
| Python | 3.12.3 |
| vLLM | 0.11.0 |
| Transformers | 4.57.1 |
| PyTorch | 2.8.0 |
| 확인한 GPU 메모리 | 16GB |

`run_vlm.sh`는 작업 폴더로 이동한 뒤 전용 Python으로 실행하며, `PYTHONPATH`와 `PYTHONHOME`을 제거하고 사용자 공용 패키지 로드를 차단합니다. 매번 가상환경을 수동으로 활성화할 필요는 없습니다.

VS Code의 Python도 `.venv-vlm/bin/python`으로 선택했습니다. 새 터미널의 패키지 경로 혼입을 막는 설정은 `.vscode/settings.json`에 저장되어 있습니다. 기존 터미널에서는 위 실행 스크립트를 사용하면 됩니다.

## 메모리와 출력 설정

- 최대 모델 길이: 2048토큰
- 동시 요청 수: 1
- 배치 토큰 상한: 2048
- 이미지: 요청당 1장, 비디오: 0개
- 이미지 전처리 픽셀 범위: 784~200704
- 기본 테스트 이미지: 최대 512×512로 축소하여 저장됨
- CUDA 그래프 비활성화(`enforce_eager=True`)
- GPU 메모리 비율 기본값: 0.90
- 출력 최대 길이: 128토큰

출력이 중간에 끊기면 `vlm_vllm_example.py`의 `SamplingParams(..., max_tokens=128)` 값을 늘립니다. 입력과 출력은 최대 모델 길이 안에 들어가야 합니다.

명령행에서 길이와 GPU 메모리 비율을 변경할 수도 있습니다.

```bash
bash run_vlm.sh --max-model-len 2048 --gpu-mem 0.85
```

## 로그 확인

화면 출력과 오류는 `vlm_run.log`에 함께 저장됩니다. **다시 실행하면 기존 로그를 덮어씁니다.** 보존할 로그는 먼저 다른 이름으로 복사합니다.

```bash
tail -n 100 vlm_run.log
cp vlm_run.log vlm_run_saved.log
```

## 파일 구성

| 파일/폴더 | 용도 |
| --- | --- |
| `vlm_vllm_example.py` | 모델 실행 및 이미지 질의 코드 |
| `run_vlm.sh` | 환경을 분리하여 실행하고 로그 저장 |
| `test_image.jpg` | 기본 테스트 이미지 |
| `requirements-vlm.txt` | 주요 패키지 버전과 설치 목록 |
| `.venv-vlm/` | 이번에 만든 전용 실행 환경 |
| `.vscode/settings.json` | VS Code Python 및 터미널 환경 설정 |
| `vlm_run.log` | 가장 최근 실행 로그 |
| `vlm_install.log` | 환경 설치 로그 |
| `vlm_vllm_example.py.before-vlm-fix` | 수정 전 원본 코드 백업 |
| `.venv/` | 기존 환경. 새 실행 스크립트에서는 사용하지 않음 |

## 문제가 생겼을 때

- `rope_type=default`와 `type=mrope` 충돌: 이전 환경 또는 Transformers 5.x를 사용 중인지 확인하고 `bash run_vlm.sh`로 실행합니다.
- `flash_attn_2_cuda`의 `undefined symbol`: 외부 패키지 경로 혼입 여부를 먼저 확인합니다. 공용 패키지를 삭제하기 전에 전용 실행 스크립트와 로그를 확인합니다.
- 이미지 HTTP 400/403 또는 파일 없음: 다운로드 가능한 주소인지, 로컬 이미지 경로가 맞는지 확인합니다.
- GPU 메모리 부족: `nvidia-smi`로 다른 프로세스 사용량을 확인합니다. 다른 사용자의 프로세스는 종료하지 않습니다.
- FlashInfer 미설치 경고: 검증한 실행에서는 PyTorch 방식으로 대체되어 정상 동작했습니다. 이 경고만으로 설치를 다시 할 필요는 없습니다.

## 설치된 환경 확인

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 .venv-vlm/bin/python -c "import sys,vllm,transformers,torch; print(sys.executable); print(vllm.__version__,transformers.__version__,torch.__version__)"
```

2026-10-27 검증: 기본 테스트 이미지에서 고양이 두 마리를 한국어로 설명했고 종료 코드 0을 확인했습니다.
