import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.models.basemodel import BaseModel, DEFAULT_TIMEOUT
from justai.models.openai_responses import OpenAIResponsesModel
from justai.tools.display import color_print, ERROR_COLOR


class XAIModel(OpenAIResponsesModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by X AI."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        keyname = 'X_API_KEY'
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values()[keyname]
        if not api_key:
            color_print('No X AI API key found. Create one at https://console.x.ai and '
                        f'set it in the .env file like {keyname}=here_comes_your_key.', color=ERROR_COLOR)
        timeout = params.get('timeout', DEFAULT_TIMEOUT)
        self.client = OpenAI(api_key=api_key, base_url='https://api.x.ai/v1', timeout=timeout)

        self.supports_image_generation = False
        self.last_response_id = None
