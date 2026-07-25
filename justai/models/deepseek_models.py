
from openai import OpenAI

from justai.models.basemodel import get_api_key, BaseModel, DEFAULT_TIMEOUT
from justai.models.openai_completions import OpenAICompletionsModel


class DeepSeekModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by DeepSeek."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, 'DEEPSEEK_API_KEY', 'DeepSeek',
                              'https://platform.deepseek.com/api_keys')
        timeout = params.get('timeout', DEFAULT_TIMEOUT)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=timeout)

        self.messages = [{"role": "system", "content": self.system_message}]

        # Overwrite parent class defaults
        self.supports_image_input = False
        self.max_output_tokens = 8192
