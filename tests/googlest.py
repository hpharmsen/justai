import json

from examples.return_types import json_example, structured_output_with_type_annotations, structured_output_with_pydantic
from justai import Model

if __name__ == '__main__':
    models = [
        Model("gemini-2.5-flash"),
        Model("gpt-5-nano"),
        Model('claude-3-7-sonnet-latest')
              ]

    # # # prompt with token usage and response time
    # for model in models:
    #     print(model.prompt('Hoeveel is twee plus twee?'))
    #     print(model.input_token_count, model.output_token_count, model.last_response_time)
    #
    # # prompt cache
    # for model in models:
    #     print(model.prompt('Hoeveel is twee plus twee?'))
    #
    # # chat
    # for model in models:
    #     model.chat('Hoeveel is twee plus twee?')
    #     print(model.chat('En als je daar nog drie bij optelt?'))
    #
    # # Images
    # for model in models:
    #     print('\n',model.model.model_name)
    #     print(model.prompt('Welke twee dieren zie je?', images=[
    #         'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg',
    #         'https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/2005-bandipur-tusker.jpg/960px-2005-bandipur-tusker.jpg'],
    #                        cached=False))
    #
    # # Return types
    # for model in models:
    #     print('\n',model.model.model_name)
    #     data = json_example(model.model.model_name)
    #     print('json')
    #     print(json.dumps(data, indent=4))
    #
    #     print('type annotations')
    #     try:
    #         data = structured_output_with_type_annotations(model.model.model_name)
    #         for person in data:
    #             print(person)
    #     except NotImplementedError as e:
    #         print('not implemented')
    #
    #     print('pydantic')
    #     try:
    #         data = structured_output_with_pydantic(model.model_name)
    #         for person in data.persons:
    #             print(person)
    #     except NotImplementedError as e:
    #         print('not implemented')

    # Async
    # see asycest.py


    # # Token count beforehand
    # prompt = 'Hoeveel is twee plus twee?'
    # for model in models:
    #     print(model.model_name, model.token_count(prompt))

    # Configuration



    # Tool use

