""" Example that shows the usage of prompt caching """

from justai import Agent
from return_types import get_story

SYSTEM_MESSAGE = """"You are a text analyzer. You answer questions about a text. 
Your answers are concise and to the point."""
MODEL = 'claude-3-5-sonnet-20240620'


def agent_test():
    agent = Agent(MODEL, max_tokens=1024)

    # First without cached prompt
    agent.system_message = SYSTEM_MESSAGE
    res = agent.chat(get_story() + 'Who is Mr. Thompsons Neighbour? Give me just the name.', 
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(agent)

    # Now with cached prompt
    agent.system_message = SYSTEM_MESSAGE
    agent.cached_prompt = get_story()
    res = agent.chat('Who is Mr. Thompsons Neighbour? Give me just the name.', 
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(agent)

    res = agent.chat('Who called it an accident? Give me just the name.', 
                     cached=False)  # Disable justais own cache
    print(res)
    show_token_usage(agent)
    

def show_token_usage(agent):
    print('input_token_count', agent.input_token_count)
    print('output_token_count', agent.output_token_count)
    print('cache_creation_input_tokens', agent.cache_creation_input_tokens)
    print('cache_read_input_tokens', agent.cache_read_input_tokens)
    print()


if __name__ == '__main__':
    agent_test()
    