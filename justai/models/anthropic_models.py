import os

import anthropic
from dotenv import load_dotenv

from justai.models.model import Model
from justai.tools.display import ERROR_COLOR, color_print

# Claude 3 Opus	    claude-3-opus-20240229
# Claude 3 Sonnet	claude-3-sonnet-20240229
# Claude 3 Haiku	Coming soon


class AnthropicModel(Model):

    def __init__(self, model_name: str, params: dict):
        """ Model implemention should create attributes for all supported parameters """
        self.model_name = model_name  # e.g. "claude-3-opus-20240229"
        self.system_message = f"You are {model_name}, a large language model trained by Anthropic."

        # Authentication
        if not os.getenv("ANTHROPIC_API_KEY"):
            load_dotenv()  # Load the .env file into the environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            color_print("No Anthropic API key found. Create one at https://console.anthropic.com/settings/keys and " +
                        "set it in the .env file like ANTHROPIC_API_KEY=here_comes_your_key.", color=ERROR_COLOR)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.debug = params.get('debug', False)

        self.max_tokens = params.get('max_tokens', 800)
        self.temperature = params.get('temperature', 0.8)

    def chat(self, prompt: str, return_json: bool, messages: list[dict], use_cache: bool = False,
             max_retries: int = 3) -> tuple[[str | object], int, int]:

        authropic_messages = [{"role": m['role'],
                               "content": [{"type": "text",
                                            "text": m['content']}]
                               } for m in messages if m['role'] != 'system']

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_message,
            messages=authropic_messages
        )
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        return message.content[0].text, input_tokens, output_tokens

    def token_count(self, text: str) -> int:
        return -1  # Not implemented
