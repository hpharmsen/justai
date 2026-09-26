"""Classification with System One models.

A System One model does not generate text. It reads a state, answers typed questions
about it, and returns calibrated probabilities. Answers arrive in tens to hundreds of
milliseconds, so each example prints the response time to make that visible.

Verified on 26-09-2026. Works via classify(), serves POST /v1/systemone:
    Jev     TypeSafe, hosted    Model('jev-latest')                     TYPESAFE_API_KEY
            via OpenRouter      Model('openrouter/typesafe/jev-1.13')   OPENROUTER_API_KEY
    Kev     Jared Palmer        Model('systemone/kev-3b', base_url=...) Apache 2.0
    Laya    Convai Innovations  via laya-serve                          Apache 2.0
    Von, Decider and other self-hosted servers that mirror the same endpoint.

Same model category, own protocol, so NOT reachable via classify():
    Tev1-4B           Together AI   chat completions with 2 to 24 options
    GLiNER2.5-Decide  Fastino       own classify_text API
    Bespoke Nimble    Bespoke Labs  own schema format

A curated list of what serves the standard lives at github.com/yanng981/awesome-system-one.
"""

import os

from dotenv import load_dotenv

from justai import Model

MODEL = 'openrouter/typesafe/jev-1.13'

TICKET = """Subject: still no refund
I cancelled on the 3rd and was charged again on the 5th. This is the fourth email I send
about this and nobody has replied. I want my money back today."""


def report(model, answer):
    print(answer)
    print(model.last_token_count(), 'tokens')
    print(f'{model.last_response_time * 1000:.0f} ms\n')


def choice_example():
    """A dict of options gives a choice: one winner plus a probability per option."""
    print('*** choice ***')
    model = Model(MODEL)
    answer = model.classify(
        TICKET,
        {'payments': 'money and payouts', 'frontend': 'UI and display', 'account': 'logging in'},
        instructions='Which team should pick this up?',
        cached=False,
    )
    report(model, answer)


def score_example():
    """An ordered list of levels gives a score: a position on that scale, with a legend."""
    print('*** score ***')
    model = Model(MODEL)
    answer = model.classify(
        TICKET,
        ['can wait', 'normal', 'now'],
        instructions='How urgent is this?',
        cached=False,
    )
    report(model, answer)
    # legend and probabilities are keyed by level, as ints
    print('winning level:', answer['legend'][round(answer['score'])], '\n')


def noul_example():
    """No options gives a noul: a single yes/no probability."""
    print('*** noul ***')
    model = Model(MODEL)
    answer = model.classify(TICKET, instructions='Is this customer angry?', cached=False)
    report(model, answer)


def batch_example():
    """Several questions in one call. The result is keyed by question name."""
    print('*** batch ***')
    model = Model(MODEL)
    answer = model.classify(
        TICKET,
        questions={
            'team': {
                'type': 'choice',
                'instructions': 'Which team?',
                'criteria': {'payments': 'money and payouts', 'frontend': 'UI and display'},
            },
            'urgency': {'type': 'score', 'instructions': 'How urgent?', 'criteria': ['low', 'medium', 'high']},
            'churn_risk': {'type': 'noul', 'instructions': 'Is this customer about to leave?'},
        },
        cached=False,
    )
    report(model, answer)


def unsupported_example():
    """Everything that is not a System One model refuses to classify."""
    print('*** unsupported ***')
    try:
        Model('claude-sonnet-5').classify(TICKET, {'a': 'first', 'b': 'second'}, instructions='Which one?')
    except NotImplementedError as e:
        print(e, '\n')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(override=True)
    choice_example()
    score_example()
    noul_example()
    batch_example()
    unsupported_example()
