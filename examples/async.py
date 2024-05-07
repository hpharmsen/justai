""" Demonstrates asynchronous use of justai """
import asyncio

from dotenv import load_dotenv

from justai import Agent

MODEL = "gpt-3.5-turbo"


async def chat(prompt):
    agent = Agent(MODEL)
    async for item in agent.chat_async(prompt):
        yield item
            
            
async def print_words(prompt):
    async for word in chat(prompt):
        print(word, end='')


if __name__ == "__main__":
    load_dotenv()
    prompt = "Give me 10 names for a juice bar that focuses senior citizens."
    asyncio.run(print_words(prompt))
