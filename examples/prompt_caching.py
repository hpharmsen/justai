""" Example that shows the usage of prompt caching """

from justai import Model
from return_types import get_story

SYSTEM_MESSAGE = """"You are a text analyzer. You answer questions about a text. 
Your answers are concise and to the point."""
MODEL = 'claude-3-5-sonnet-20240620'


def model_test():
    model = Model(MODEL, max_tokens=1024)

    # First without cached prompt
    model.system_message = SYSTEM_MESSAGE
    res = model.chat(get_story() + 'Who is Mr. Thompsons Neighbour? Give me just the name.',
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)

    # Now with cached prompt
    model.system_message = SYSTEM_MESSAGE
    model.cached_prompt = get_story()
    res = model.chat('Who is Mr. Thompsons Neighbour? Give me just the name.',
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)

    res = model.chat('Who called it an accident? Give me just the name.',
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)
    

def show_token_usage(model):
    print('input_token_count', model.input_token_count)
    print('output_token_count', model.output_token_count)
    print('cache_creation_input_tokens', model.cache_creation_input_tokens)
    print('cache_read_input_tokens', model.cache_read_input_tokens)
    print()


if __name__ == '__main__':
    model_test()
    