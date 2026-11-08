"""
vLLM으로 VLM(Vision-Language Model) 돌려보기 예제

필요 환경
- Linux + NVIDIA GPU (vLLM은 기본적으로 Linux/CUDA 환경 기준. Windows라면 WSL2 권장)
- pip install vllm pillow requests

실행
    python vlm_vllm_example.py
    python vlm_vllm_example.py --image ./my_photo.jpg --prompt "이 이미지를 설명해줘"
"""

import argparse
import base64
import mimetypes
from pathlib import Path

from vllm import LLM, SamplingParams

# GPU 메모리가 작다면 더 작은 모델(예: Qwen/Qwen2-VL-2B-Instruct)로 바꿔보세요.
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # 16GB GPU용 (7B는 bf16 가중치만 ~16GB라 OOM)
DEFAULT_IMAGE_URL = str(Path(__file__).resolve().with_name("test_image.jpg"))


def to_image_url(image: str) -> str:
    """URL이면 그대로, 로컬 파일이면 base64 data URL로 변환."""
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL, help="이미지 URL 또는 로컬 경로")
    parser.add_argument("--prompt", default="이 이미지에 무엇이 보이는지 한국어로 자세히 설명해줘.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="gpu_memory_utilization")
    args = parser.parse_args()

    # 1) 모델 로드
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,        # 줄이면 메모리 절약
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"image": 1, "video": 0},
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        enforce_eager=True,
        mm_processor_kwargs={"min_pixels": 784, "max_pixels": 200704},        # 프롬프트당 이미지 개수 제한
    )

    # 2) 샘플링 파라미터
    sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=512)

    # 3) OpenAI 스타일 chat 메시지 (이미지 + 텍스트)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_image_url(args.image)}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # 4) 추론 (chat 템플릿은 vLLM이 모델에 맞게 자동 적용)
    outputs = llm.chat(messages, sampling_params=sampling_params)

    print("\n===== 결과 =====")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# [참고] 서버 모드로 띄우고 OpenAI 클라이언트로 호출하고 싶다면
#
# 1) 터미널에서 서버 실행
#    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192
#
# 2) 파이썬 클라이언트 (pip install openai)
#    from openai import OpenAI
#    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
#    resp = client.chat.completions.create(

#        model="Qwen/Qwen2.5-VL-7B-Instruct",
#        messages=[{"role": "user", "content": [
#            {"type": "image_url", "image_url": {"url": "https://.../image.jpg"}},
#            {"type": "text", "text": "이미지를 설명해줘"},
#        ]}],
#        max_tokens=128,
#    )
#    print(resp.choices[0].message.content)
# ---------------------------------------------------------------------------