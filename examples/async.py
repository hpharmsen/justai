""" Demonstrates asynchronous use of justai """
import asyncio

from dotenv import load_dotenv

from justai import Agent

async def chat(prompt):
    agent = Agent("deepseek-reasoner")
    agent = Agent("gemini-1.5-flash")
    async for word, reasoning_content in agent.chat_async(prompt):
        yield word, reasoning_content
            
            
async def print_words(prompt):
    async for word, reasoning_content in chat(prompt):
        if reasoning_content:
            print(reasoning_content, end='-') # Using a dash here to show what is reasoning content
        if word:
            print(word, end='')


if __name__ == "__main__":
    load_dotenv()
    prompt = "Give me 5 names for a juice bar that focuses senior citizens."
    asyncio.run(print_words(prompt))
