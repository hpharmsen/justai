import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.models.basemodel import BaseModel
from justai.models.openai_models import OpenAIModel
from justai.tools.display import color_print, ERROR_COLOR


class DeepSeekModel(OpenAIModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by DeepSeek."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        keyname = "DEEPSEEK_API_KEY"
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values()[keyname]
        if not api_key:
            color_print("No DEEPSEEK API key found. Create one at https://platform.deepseek.com/api_keys and " +
                        f"set it in the .env file like {keyname}=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
