""" Demonstrates asynchronous use of justai """
import asyncio

from dotenv import load_dotenv

from justai import Agent

async def chat(prompt):
    agent = Agent("deepseek-reasoner")
    async for item in agent.chat_async(prompt):
        yield item
            
            
async def print_words(prompt):
    async for word in chat(prompt):
        print(word, end='')


if __name__ == "__main__":
    load_dotenv()
    prompt = "Give me 5 names for a juice bar that focuses senior citizens."
    asyncio.run(print_words(prompt))
