import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.models.model import Model
from justai.models.openai_models import OpenAIModel
from justai.tools.display import color_print, ERROR_COLOR


class XAIModel(OpenAIModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by X AI."
        Model.__init__(self, model_name, params, system_message)

        # Authentication
        api_key = params.get("X_API_KEY") or os.getenv("X_API_KEY") or dotenv_values()["X_API_KEY"]
        if not api_key:
            color_print("No X AI API key found. Create one at https://console.x.ai and " +
                        "set it in the .env file like X_API_KEY=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
