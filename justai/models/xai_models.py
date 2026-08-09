import re
from typing import Any

from openai import OpenAI

from justai.models.basemodel import get_api_key, BaseModel, client_retry_kwargs, client_timeout
from justai.models.openai_responses import OpenAIResponsesModel


# Grok reasoning models (mid-2026): grok-4.5, grok-4.3, grok-4.20-multi-agent.
EFFORT_MODELS_GROK = re.compile(r'grok-4\.(5|3|20)')

_EFFORT_MAP_GROK = {
    'low': ('low', None),
    'medium': ('medium', None),
    'high': ('high', None),
    'xhigh': ('high', 'xhigh -> high (Grok has no higher tier)'),
    'max': ('high', 'max -> high (Grok has no higher tier)'),
}


class XAIModel(OpenAIResponsesModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by X AI."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, 'X_API_KEY', 'X AI', 'https://console.x.ai')
        self.client = OpenAI(api_key=api_key, base_url='https://api.x.ai/v1', timeout=client_timeout(params), **client_retry_kwargs(params))

        self.supports_image_generation = False
        self.last_response_id = None

    def _validate_effort(self, value: Any) -> None:
        # xAI does not accept OpenAI's 'none' pass-through; use the base validation only.
        BaseModel._validate_effort(self, value)

    def resolve_effort(self) -> tuple[str | None, str | None]:
        level = self.model_params.get('effort')
        if level is None:
            return (None, None)
        if EFFORT_MODELS_GROK.search(self.model_name):
            native, warn_key = _EFFORT_MAP_GROK[level]
            warn = None if warn_key is None else f'effort={level!r} not natively supported by {self.model_name}; {warn_key}'
            return (native, warn)
        return (None, f'effort is not supported by {self.model_name}, ignoring')
