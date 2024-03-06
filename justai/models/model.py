from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def __init__(self, model_name: str, params: dict):
        """ Model implemention should create attributes for all supported parameters """
        self.system_message = ''

    def set(self, key: str, value):
        if not hasattr(self, key):
            raise (AttributeError(f"Model has no attribute {key}"))
        setattr(self, key, value)

    @abstractmethod
    def chat(self, prompt: str, return_json: bool, messages: list[dict], use_cache: bool = False,
             max_retries: int = 3) -> tuple[[str | object], int, int]:
        pass

    @abstractmethod
    def token_count(self, text: str) -> int:
        pass
