import io
import httpx

from PIL import Image
from justai import Agent


if __name__ == "__main__":
    # Example with image from url and Gemini
    agent = Agent("gemini-1.5-flash")
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/1928_Model_A_Ford.jpg/520px-1928_Model_A_Ford.jpg'
    message = agent.chat("What is in this image", images=url, cached=False)
    print(message)

    # Example with raw image data and Claude
    agent = Agent("claude-3-sonnet-20240229")
    url = 'https://upload.wikimedia.org/wikipedia/commons/5/5a/Betula_pendula_winter.jpg'
    img_data = httpx.get(url).content
    message = agent.chat("What is in this image", images=img_data, cached=False)
    print(message)

    # Example with PIL image and GPT4o
    agent = Agent("gpt-4o")
    url = 'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg'
    image = Image.open(io.BytesIO(httpx.get(url).content))
    message = agent.chat("What is in this image", images=url, cached=False)
    print(message)
    print(agent.last_token_count(), 'tokens')  # (input_token_count, output_token_count, total_token_count)
