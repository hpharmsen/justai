import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.model.message import Message
from justai.models.basemodel import BaseModel
from justai.models.openai_models import OpenAIModel
from justai.tools.display import color_print, ERROR_COLOR


class PerplexityModel(OpenAIModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by Perplexity."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        keyname = "PERPLEXITY_API_KEY"
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values()[keyname]
        if not api_key:
            color_print(f"No {keyname} found. Create one at https://www.perplexity.ai/settings/api and " +
                        f"set it in the .env file like {keyname}=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    def chat_async(self, messages: list[Message]):
        """ Perplexity does not separately return thinking content
        but returns it's thinking between <think> and </think> tags."""
        thinking = False
        for content, _ in super().chat_async(messages):
            if content == '<think>':
                thinking = True
            elif '</think>' in content:
                thinking = False
            elif thinking:
                yield None, content
            else:
                yield content, None
