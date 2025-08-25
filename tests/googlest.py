import json

from dotenv import load_dotenv

from examples.prompt_caching import caching_example
from examples.return_types import json_example, structured_output_with_pydantic
from justai import Model

if __name__ == '__main__':
    load_dotenv()
    models = [
        Model("gpt-5-nano", temperature=1),
        Model('claude-3-7-sonnet-latest', temperature=1),
        Model("gemini-2.5-flash", temperature=1),
        Model("grok-4", temperature=0),
        Model("deepseek-chat", temperature=0),
        Model("sonar", temperature=0),
        Model("openrouter/anthropic/claude-3.7-sonnet", temperature=0),
    ]

    # # prompt with token usage and response time
    # for model in models:
    #     model.system = "You are a calculator.You always answer in binary format"
    #     print(model.prompt('Hoeveel is twee plus twee?', cached=False))
    #     print(model.input_token_count, model.output_token_count, model.last_response_time)
    #     model.system = ""
    #
    # # prompt cache
    # for model in models:
    #     print(model.prompt('Hoeveel is twee plus twee?', cached=True))
    #     print(model.input_token_count, model.output_token_count, model.last_response_time)
    #     print(model.prompt('Hoeveel is twee plus twee?', cached=True))
    #     print(model.input_token_count, model.output_token_count, model.last_response_time)
    #
    # # chat
    # for model in models:
    #     try:
    #         model.chat('Hoeveel is twee plus twee?')
    #         print(model.chat('En als je daar nog drie bij optelt?'))
    #     except NotImplementedError as e:
    #         print(e)
    #
    # # Images
    # for model in models:
    #     print('\n',model.model.model_name)
    #     try:
    #         print(model.prompt('Welke twee dieren zie je?', images=[
    #             'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg',
    #             'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/2005-bandipur-tusker.jpg/960px-2005-bandipur-tusker.jpg'],
    #                            cached=False))
    #     except NotImplementedError as e:
    #         print(e)

    # Return types
    for model in models:
        print('\n',model.model_name)
        try:
            data = json_example(model)
            print('json')
            print(json.dumps(data, indent=4))
        except NotImplementedError as e:
            print(e)

        print('pydantic')
        try:
            data = structured_output_with_pydantic(model)
            for person in data.persons:
                print(person)
        except NotImplementedError as e:
            print(e)
    #
    # Async
    # see asycest.py


    # Token count beforehand
    prompt = 'Hoeveel is twee plus twee?'
    for model in models:
        print(model.model_name, model.token_count(prompt), 'tokens')

    # Configuration
    # Done


    # Tool use
    def ergobam(x: int, y: int) -> int:
        """Fictional function that does nothing useful"""
        return 27

    def kwaroot(x: int) -> int:
        """Fictional function that does nothing useful"""
        return x // 3

    for model in models:
        print("\n", model.model.model_name)
        try:
            if model.model.supports_automatic_function_calling:
                model.add_tool(ergobam)
                model.add_tool(kwaroot)
            else:
                model.add_tool(ergobam, description="Calculates the ergobam of two numbers",
                               parameters={"x": int, "y": int}, required_parameters=["x", "y"])
                model.add_tool(kwaroot, description="Calculates the kwaroot of an integer",
                               parameters={"x": int}, required_parameters=["x"])
        except NotImplementedError as e:
            print(e)
            continue
        try:
            print(model.prompt("Give me the kwaroot of the ergobam of 3 and 4.", cached=False))
        except NotImplementedError as e:
            print(e)
        pass

    # Content caching
    for model in models:
        if  model.model.supports_cached_prompts:
            print("\n", model.model.model_name)
            caching_example(model)

    # Generate images

