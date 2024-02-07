import sys

sys.path.append('.')
from justai import Agent


if __name__ == "__main__":
    agent = Agent('gpt-3.5-turbo')
    agent.system = lambda: "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."

    message = agent.chat("Forrest Gump")
    print(message)
