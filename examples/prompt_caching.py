""" Example that shows the usage of prompt caching """

from justai import Model
from examples.return_types import get_story

SYSTEM_MESSAGE = """"You are a text analyzer. You answer questions about a text. 
Your answers are concise and to the point."""


def caching_example(model: Model):

    # First without cached prompt
    model.system_message = SYSTEM_MESSAGE
    res = model.prompt(get_story() + 'Who is Mr. Thompsons Neighbour? Give me just the name.',
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)

    # Now with cached prompt
    model.system_message = SYSTEM_MESSAGE
    model.cached_prompt = get_story()
    res = model.prompt('Who is Mr. Thompsons Neighbour? Give me just the name.',
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)

    res = model.prompt('Who called it an accident? Give me just the name.',
                       cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(model)
    

def show_token_usage(model):
    print('input_token_count', model.input_token_count)
    print('output_token_count', model.output_token_count)
    if hasattr(model, 'cache_read_input_tokens'):
        print('cache_creation_input_tokens', model.cache_creation_input_tokens)
    if hasattr(model, 'cache_read_input_tokens'):
        print('cache_read_input_tokens', model.cache_read_input_tokens)
    print()


if __name__ == '__main__':
    model = Model('claude-3-7-sonnet-latest')
    caching_example(model)
    