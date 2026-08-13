from openai import OpenAI

from justai.models.basemodel import get_api_key, BaseModel, client_retry_kwargs, client_timeout
from justai.models.openai_completions import OpenAICompletionsModel


class KimiModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name}, a large language model trained by Moonshot AI.'
        BaseModel.__init__(self, model_name, params, system_message)

        api_key = get_api_key(params, 'MOONSHOT_API_KEY', 'Moonshot', 'https://platform.moonshot.ai/')
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.moonshot.ai/v1',
            timeout=client_timeout(params),
            **client_retry_kwargs(params),
        )

        self.messages = [{'role': 'system', 'content': self.system_message}]

        self.supports_function_calling = True
        self.supports_image_input = True
        self.max_output_tokens = 8192

        # Kimi rejects temperature > 1 (OpenAI accepts [0, 2]). Clamp to a safe value.
        if self.model_params.get('temperature', 0) > 1:
            self.model_params['temperature'] = 1.0
