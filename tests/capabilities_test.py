"""Test JustAI capabilities with various AI providers.

Capabilities tested: async, json, pydantic, vision, tooluse

Usage:
    python tests/capabilities_test.py                           # All tests, all models
    python tests/capabilities_test.py -m gpt-5-mini gemini-2.5-flash
    python tests/capabilities_test.py -t async pydantic
    python tests/capabilities_test.py -t json -m deepseek-chat sonar
"""
import argparse
import asyncio

from dotenv import load_dotenv

from justai import Model

from examples.return_types import json_example, structured_output_with_pydantic

ALL_MODELS = [
    'gpt-5-mini',
    'claude-haiku-4-5',
    'gemini-2.5-flash',
    'grok-4-fast-non-reasoning',
    'deepseek-chat',
    'sonar',
]


def run_async(model_name):
    """Test async streaming."""
    async def run_prompt():
        model = Model(model_name)
        async for _ in model.prompt_async('Say hello'):
            pass
    asyncio.run(run_prompt())


def run_json(model_name):
    """Test JSON response format."""
    model = Model(model_name)
    json_example(model)


def run_pydantic(model_name):
    """Test structured output with Pydantic."""
    model = Model(model_name)
    structured_output_with_pydantic(model)


def run_vision(model_name):
    """Test image/vision input."""
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/1928_Model_A_Ford.jpg/520px-1928_Model_A_Ford.jpg'
    model = Model(model_name)
    model.chat('What is in this image', images=url, cached=False)


def run_tooluse(model_name):
    """Test tool/function calling."""
    def get_temperature(city: str) -> str:
        temps = {'amsterdam': '18', 'london': '15', 'paris': '20', 'berlin': '17'}
        return temps.get(city.lower(), '22')

    model = Model(model_name)
    model.add_tool(
        get_temperature,
        description='Get the current temperature for a city',
        parameters={'city': str},
        required_parameters=['city']
    )
    response = model.prompt('What is the temperature in Amsterdam?', cached=False)
    assert '18' in str(response), f'Tool not called correctly: {response[:100]}'


ALL_TESTS = {
    'async': run_async,
    'json': run_json,
    'pydantic': run_pydantic,
    'vision': run_vision,
    'tooluse': run_tooluse,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test JustAI capabilities')
    parser.add_argument('-m', '--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS, metavar='MODEL',
                        help=f'Models to test. Available: {", ".join(ALL_MODELS)}')
    parser.add_argument('-t', '--tests', nargs='+', default=list(ALL_TESTS.keys()),
                        choices=ALL_TESTS.keys(), metavar='TEST',
                        help=f'Tests to run. Available: {", ".join(ALL_TESTS.keys())}')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv(override=True)
    args = parse_args()

    for model_name in args.models:
        print(f'===================== {model_name} ' + '=' * (50 - len(model_name)))
        for test_name in args.tests:
            try:
                ALL_TESTS[test_name](model_name)
                print(f'{test_name}, ok')
            except NotImplementedError as e:
                print(f'{test_name}, not supported: {e}')
            except Exception as e:
                print(f'{test_name}, error: {e}')
