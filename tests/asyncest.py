#!/usr/bin/env python3
import os
import sys
import asyncio
from dotenv import load_dotenv
from justai import Model

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'


async def main():
    load_dotenv()

    models = [
        Model("gpt-5-nano", temperature=1),
        #Model('claude-3-7-sonnet-latest', temperature=1),
        Model("gemini-2.5-flash", temperature=1),
        Model("grok-4", temperature=0),
        Model("deepseek-chat", temperature=0),
        Model("sonar", temperature=0),
        Model("openrouter/anthropic/claude-3.7-sonnet", temperature=0),
    ]

    prompt = "Tell me a short story about a robot learning to paint. Max 100 words."

    for model in models:
        print(f"\n\nUsing model: {model.model.model_name}")
        async for content, _ in model.prompt_async(prompt):
            if content:
                print(content, end='', flush=True)
        print()  # Add newline at the end


if __name__ == '__main__':
    asyncio.run(main())