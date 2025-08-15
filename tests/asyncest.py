#!/usr/bin/env python3
import os
import sys
import asyncio
from dotenv import load_dotenv
from justai import Model

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

models = [
    # Model("gemini-2.5-flash"),
    # Model("gpt-5-nano"),
    Model('claude-3-7-sonnet-latest')
]

async def main():
    load_dotenv()
    
    prompt = "Tell me a short story about a robot learning to paint. Max 100 words."

    for model in models:
        print(f"\n\nUsing model: {model.model.model_name}")
        async for content, _ in model.prompt_async(prompt):
            if content:
                print(content, end='', flush=True)
        print()  # Add newline at the end


if __name__ == '__main__':
    asyncio.run(main())