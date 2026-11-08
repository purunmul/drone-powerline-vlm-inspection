# 전선 검출 코드 사용법

학교 서버 작업 폴더는 `/home/work/123-0125`입니다. 아래 명령은 VS Code 터미널에서 이 폴더를 기준으로 실행합니다.

## 1. 현재 결과 보기

`results/vlm_zoom_recheck_round2_v1/`에 개발 사진 3장의 1차 시각화 결과가 있습니다.

- `<사진 ID>/prediction_on_original.jpg` — 원본 크기 3840×2160에 예측 전선을 청록색으로 표시한 이미지
- `<사진 ID>/prediction_centerline.png` — 평가에 사용한 1픽셀 중심선
- `preview.jpg` — 미리보기
- `metrics.csv`, `summary.json` — 사진별 지표와 조건별 평균
- `results/vlm_zoom_recheck_round2_v1.zip` — 위 결과의 다운로드용 묶음

채택한 흐름은 **원본 사진 → LangSAM 기본 검출 → 고정 규칙으로 고른 4개 구역 확대 재검출 → 마스크별 중심선 추출·통합 → 원본 위에 표시**입니다. Qwen의 구역 선택은 별도 비교 실험이었습니다. XML에서 만든 정답 중심선은 평가에만 사용합니다.

## 2. 현재 결과 재현하기 · GPU 없이 실행

저장된 LangSAM 기본 중심선과 확대 마스크로 비교 실험과 결과 이미지를 재현합니다. 기본 명령은 `results/reproductions/vlm_zoom_recheck_round2_v1/`과 같은 이름의 ZIP을 새로 만들며, 기존 결과는 보존됩니다.

```bash
cd /home/work/123-0125
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-eval/bin/python -u pipeline/evaluate_vlm_zoom_recheck_round2_v1.py
```

같은 명령을 다시 실행하려면 끝에 `--output-dir results/reproductions/check_02`처럼 아직 없는 경로를 지정합니다. 저장된 입력은 `results/vlm_zoom_recheck_round1_v1/`과 `results/skeleton_order_ablation_20261107_v1/`에 있습니다. 실행 로그에 `VERIFIED`가 나오면 이미지 묶음 검사를 통과한 것입니다. `summary.json`의 수치는 개발용 3장 평균이며, 독립 평가 점수와 구분합니다.

## 3. 새 사진에서 LangSAM 후보 검출 · GPU 필요

아래 두 명령은 새 사진의 전체 장면 검출과 겹치는 타일 검출까지 수행합니다. 예시의 이미지 경로와 출력 폴더명은 실제 파일에 맞게 바꾸세요.

```bash
cd /home/work/123-0125
bash pipeline/run_langsam.sh pipeline/run_langsam_baseline.py \
  --image data/my_photo.jpg --output results/new_image/whole
bash pipeline/run_langsam.sh pipeline/run_langsam_tiled.py \
  --baseline results/new_image/whole/result.json --output results/new_image/tiles \
  --tile-size 1024 --overlap 256
```

검출 후보의 좌표와 마스크는 각 출력 폴더에 저장됩니다. 2번의 재현 스크립트는 세 개발 사진과 저장된 확대 결과를 사용합니다. 임의의 새 사진을 넣어 확대 구역 선택부터 최종 이미지까지 자동 실행하는 진입점은 아직 없습니다.

## 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `pipeline/` | 현재 재현 코드, LangSAM 실행 코드, 재현에 필요한 공통 코드 |
| `data/` | 원본 사진과 XML |
| `results/` | 현재 결과와 재현·관련 실험에 쓰이는 입력 |
| `archive/experiments_20261108/` | 이전 실험 코드와 결과, 로그, ZIP, 기존 사용법 원본 |

보관본의 `move_manifest.tsv`에서 각 항목의 원래 경로를 확인할 수 있습니다. 일부 이전 실험은 그 경로를 기준으로 작성돼 있습니다. `.venv-eval`, `.venv-langsam`, 모델 캐시와 `vendor/`는 현재 서버에서 사용 중인 설치 환경입니다.
