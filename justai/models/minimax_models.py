from openai import OpenAI

from justai.models.basemodel import get_api_key, BaseModel, client_retry_kwargs, client_timeout
from justai.models.openai_completions import OpenAICompletionsModel


class MiniMaxModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name}, a large language model trained by MiniMax.'
        BaseModel.__init__(self, model_name, params, system_message)

        api_key = get_api_key(params, 'MINIMAX_API_KEY', 'MiniMax', 'https://platform.minimax.io/')
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.minimax.io/v1',
            timeout=client_timeout(params),
            **client_retry_kwargs(params),
        )

        self.messages = [{'role': 'system', 'content': self.system_message}]

        self.supports_function_calling = True
        # Only the M3 family takes image input. M2.x accepts an image_url part without erroring
        # and then answers as if it saw nothing, so refuse up front rather than return a bogus answer.
        self.supports_image_input = model_name.lower().startswith('minimax-m3')
        self.max_output_tokens = 512_000 if self.supports_image_input else 131_072

    def _extra_api_kwargs(self) -> dict:
        # Without this MiniMax inlines its chain of thought in message.content as <think>...</think>,
        # which corrupts both plain answers and JSON parsing.
        return {'extra_body': {'reasoning_split': True}}
