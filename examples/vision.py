import os
import httpx

import openai
from dotenv import load_dotenv
from justai import Agent


if __name__ == "__main__":
    load_dotenv()  # Load the .env file into the environment
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Example with raw image data and Claude
    agent = Agent("claude-3-sonnet-20240229")
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/1928_Model_A_Ford.jpg/520px-1928_Model_A_Ford.jpg'
    img_data = httpx.get(url).content
    message = agent.chat("What is in this image", image=img_data)
    print(message)

    # Example with image from url and GPT
    agent = Agent("gpt-4o")
    url = 'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg'
    message = agent.chat("What is in this image", image=url)
    print(message)
    print(agent.last_token_count(), 'tokens')  # (input_token_count, output_token_count, total_token_count)
