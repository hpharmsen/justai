""" Demonstrates asynchronous use of justai """
import asyncio

from dotenv import load_dotenv

from justai import Model


async def print_words(model_name, prompt):
    model = Model(model_name)
    async for word, reasoning_content in model.chat_async_reasoning(prompt):
        if reasoning_content:
            print(reasoning_content, end='-') # Using a dash here to show what is reasoning content
        if word:
            print(word, end='')


async def print_words_reasoning(model_name, prompt):
    model = Model(model_name)
    async for word, reasoning_content in model.chat_async_reasoning(prompt):
        if reasoning_content:
            print(reasoning_content, end='-') # Using a dash here to show what is reasoning content
        if word:
            print(word, end='')


if __name__ == "__main__":
    load_dotenv()
    prompt = "Give me 5 names for a juice bar that focuses senior citizens."

    # Once plain
    asyncio.run(print_words("sonar-pro", prompt))

    # And once with reasoning output
    asyncio.run(print_words_reasoning("sonar-reasoning", prompt))
