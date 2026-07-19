import os

from dotenv import dotenv_values
from openai import OpenAI

from justai.models.basemodel import BaseModel, DEFAULT_TIMEOUT
from justai.models.openai_completions import OpenAICompletionsModel
from justai.tools.display import color_print, ERROR_COLOR


class KimiModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name}, a large language model trained by Moonshot AI.'
        BaseModel.__init__(self, model_name, params, system_message)

        keyname = 'MOONSHOT_API_KEY'
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values().get(keyname)
        if not api_key:
            color_print(
                'No Moonshot API key found. Create one at https://platform.moonshot.ai/ and '
                f'set it in the .env file like {keyname}=here_comes_your_key.',
                color=ERROR_COLOR,
            )
        timeout = params.get('timeout', DEFAULT_TIMEOUT)
        self.client = OpenAI(api_key=api_key, base_url='https://api.moonshot.ai/v1', timeout=timeout)

        self.messages = [{'role': 'system', 'content': self.system_message}]

        self.supports_function_calling = True
        self.supports_image_input = True
        self.max_output_tokens = 8192

        # Kimi rejects temperature > 1 (OpenAI accepts [0, 2]). Clamp to a safe value.
        if self.model_params.get('temperature', 0) > 1:
            self.model_params['temperature'] = 1.0
