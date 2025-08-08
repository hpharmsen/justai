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

models = {
    "OpenAi": "gpt-5-nano",
    "Anthropic": "claude-sonnet-4-0",
    "Google": "gemini-2.5-pro",
    "X-AI": "grok-4",
    "DeepSeek": "deepseek-chat",
    "Perplexity": "sonar-pro",
}


def run_async(model_name):
    async def run_prompt():
        prompt = "Give me 5 names for a juice bar that focuses senior citizens."
        model = Model(model_name)
        async for word in model.prompt_async(prompt):
            pass

    asyncio.run(run_prompt())
    print("Async, ok")


def run_json(model_name):
    try:
        res = json_example(model_name)
        print('Json, ok')
    except Exception as e:
        print(e)


def run_type_annotates(model_name):
    try:
        res = structured_output_with_type_annotations(model_name)
        print('Type annotations, ok')
    except Exception as e:
        print(e)


def run_pydantic(model_name):
    try:
        res = structured_output_with_pydantic(model_name)
        print('Pydantic, ok')
    except Exception as e:
        print(e)

def run_vision(model_name):
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/1928_Model_A_Ford.jpg/520px-1928_Model_A_Ford.jpg'
    model = Model(model_name)
    try:
        message = model.chat("What is in this image", images=url, cached=False)
        print("Vison, ok.", message[:80])
    except Exception as e:
        print(e)

def run_tool_use(model_name):
    print('NOT YET IMPLEMENTED')

if __name__ == "__main__":
    for provider, model_name in models.items():
        print(f"===================== {provider} - {model_name}", "=" * (50-len(provider) - len(model_name)))
        run_async(model_name)
        run_json(model_name)
        run_type_annotates(model_name)
        run_pydantic(model_name)
        run_vision(model_name)