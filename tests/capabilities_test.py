""" This Justai with the various AI providers

Capabilities that are tested are:
- Async chat
- Return JSON
- Structured types
- Token counter
- Image support
- Tool use

Providers and their key model that are tested are:
"""
import asyncio

from justai import Model

from examples.return_types import json_example, structured_output_with_pydantic, structured_output_with_type_annotations

models = {'OpenAi': 'gpt-4o',
          'Anthropic': 'claude-sonnet-4-0',
          'Google': 'gemini-2.5-pro',
          'X-AI': 'grok-4',
          'DeepSeek': 'deepseek-chat',
          'Perplexity': 'sonar-pro'}


def run_async(model_name):
    async def print_words():
        prompt = "Give me 5 names for a juice bar that focuses senior citizens."
        model = Model(model_name)
        async for word in model.prompt_async(prompt):
            print(word, end="")

    asyncio.run(print_words())

def run_json(model_name):
    json_example(model_name)

if __name__ == "__main__":
    run_json('sonar-pro')
    for provider, model_name in models.items():
        print(f"===================== {provider} - {model_name} =====================")
        #run_async(model_name)
        run_json(model_name)