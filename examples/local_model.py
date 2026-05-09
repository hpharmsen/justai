"""Run a local GGUF model with JustAI, including vision/image support.

Prerequisites:
    uv sync --extra llama

    # On Apple Silicon, rebuild with Metal support for GPU acceleration:
    CMAKE_ARGS="-DGGML_METAL=on" uv pip install --force-reinstall --no-cache-dir llama-cpp-python

    # Download a text-only model:
    huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF \
        --include "qwen2-0_5b-instruct-q5_k_m.gguf" --local-dir ./models/

    # Download a vision model (LLaVA 1.5 7B):
    huggingface-cli download mys/ggml_llava-v1.5-7b \
        --include "ggml-model-q4_k.gguf" "mmproj-model-f16.gguf" --local-dir ./models/
"""

import asyncio
import os
from pathlib import Path

from PIL import Image

from justai import Model

TEXT_MODEL = './models/qwen2-0_5b-instruct-q5_k_m.gguf'
VISION_MODEL = './models/llava-v1.5-7b-Q4_K_M.gguf'
VISION_CLIP = './models/llava-v1.5-7b-mmproj-model-f16.gguf'


def basic_prompt():
    """Single prompt, no conversation history."""
    model = Model(TEXT_MODEL)
    result = model.prompt('What is the capital of France? Answer briefly.')
    print(f'Prompt: {result}')
    print(f'Tokens (in, out, total): {model.last_token_count()}')


def multi_turn_chat():
    """Multi-turn conversation with history."""
    model = Model(TEXT_MODEL)
    print(f'Chat 1: {model.chat("My name is Alice.")}')
    print(f'Chat 2: {model.chat("What is my name?")}')


def json_response():
    """Get structured JSON output."""
    model = Model(TEXT_MODEL)
    result = model.prompt(
        'Return a JSON object with keys "city" and "country" for the capital of France.',
        return_json=True
    )
    print(f'JSON: {result}')


async def streaming():
    """Stream tokens as they are generated."""
    model = Model(TEXT_MODEL)
    print('Streaming: ', end='')
    async for content, _ in model.prompt_async('Write a haiku about coding.'):
        print(content, end='', flush=True)
    print()


def describe_image():
    """Send an image to a vision-capable local model."""
    model = Model(VISION_MODEL, clip_model_path=VISION_CLIP)
    # Create a simple test image
    img = Image.new('RGB', (200, 200), color='red')
    result = model.prompt('Describe this image. What color is it?', images=img)
    print(f'Vision: {result}')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent.parent)

    print('=== Basic Prompt ===')
    basic_prompt()

    print('\n=== Multi-turn Chat ===')
    multi_turn_chat()

    print('\n=== JSON Response ===')
    json_response()

    print('\n=== Streaming ===')
    asyncio.run(streaming())

    print('\n=== Vision (Image Description) ===')
    describe_image()
