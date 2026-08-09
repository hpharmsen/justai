
from openai import OpenAI

from justai.models.basemodel import get_api_key, BaseModel, client_retry_kwargs, client_timeout
from justai.models.openai_completions import OpenAICompletionsModel


class PerplexityModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by Perplexity."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, 'PERPLEXITY_API_KEY', 'Perplexity',
                              'https://www.perplexity.ai/settings/api')
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai", timeout=client_timeout(params), **client_retry_kwargs(params))

        self.messages = [{"role": "system", "content": self.system_message}]

        # Overwrite parent class defaults
        self.supports_return_json = False

    async def chat_async(self, prompt: str, images=None):
        """Perplexity does not separately return thinking content but returns it between <think> and </think> tags."""
        thinking = False
        async for content, _ in super().chat_async(prompt, images):
            if content == '<think>':
                thinking = True
            elif '</think>' in content:
                thinking = False
            elif thinking:
                yield None, content
            else:
                yield content, None
