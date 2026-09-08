"""Example that shows how prompt caching pays off, and how to see whether it did.

Three scenarios, each printing the cache counters:

  1. one-shot, everything in the prompt: nothing to reuse
  2. one-shot with `cached_prompt`: the fixed text moves into the cached prefix
  3. multi-turn chat: the conversation itself is cached from the second turn on

Run with a real ANTHROPIC_API_KEY. `cached=False` disables justai's own local
response cache, which would otherwise skip the API call entirely and leave the
provider-side counters at zero.
"""

from justai import Model
from examples.return_types import get_story

SYSTEM_MESSAGE = """You are a text analyzer. You answer questions about a text.
Your answers are concise and to the point."""


def one_shot_examples(model: Model):
    print('--- 1. alles in de prompt, niets te hergebruiken')
    model.system_message = SYSTEM_MESSAGE
    print(model.prompt(get_story() + 'Who is Mr. Thompsons neighbour? Just the name.', cached=False))
    show_token_usage(model)

    print('--- 2. de vaste tekst als cached_prompt')
    model.cached_prompt = get_story()
    print(model.prompt('Who is Mr. Thompsons neighbour? Just the name.', cached=False))
    show_token_usage(model)

    print('--- 3. tweede vraag op dezelfde cached_prompt: nu een read')
    print(model.prompt('Who called it an accident? Just the name.', cached=False))
    show_token_usage(model)


def multi_turn_example(model: Model):
    """De geschiedenis krijgt vanaf de tweede beurt een breakpoint.

    Beurt 1 schrijft alleen het systeemdeel. Beurt 2 schrijft de geschiedenis en
    leest het systeemdeel. Vanaf beurt 3 wordt ook de geschiedenis gelezen.
    """
    print('=== multi-turn')
    model.reset()
    model.system_message = SYSTEM_MESSAGE
    model.cached_prompt = get_story()
    for turn, question in enumerate(
        [
            'Who is Mr. Thompsons neighbour? Just the name.',
            'And who called it an accident?',
            'Summarise both answers in one sentence.',
        ],
        start=1,
    ):
        print(f'--- beurt {turn}: {question}')
        print(model.chat(question))
        show_token_usage(model)


def show_token_usage(model):
    read = model.cache_read_input_tokens
    written = model.cache_creation_input_tokens
    # Bij Anthropic staan deze tokens los van input_tokens, niet erin.
    total_prompt = model.input_token_count + read + written
    print(f'  input (vol tarief) {model.input_token_count}')
    print(f'  cache write (1,25x) {written}')
    print(f'  cache read (0,1x)   {read}')
    if total_prompt:
        print(f'  hit rate {read / total_prompt:.0%} van {total_prompt} prompt-tokens')
    print(f'  output {model.output_token_count}')
    print()


if __name__ == '__main__':
    one_shot_examples(Model('claude-sonnet-4-6'))
    multi_turn_example(Model('claude-sonnet-4-6'))
