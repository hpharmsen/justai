from abc import ABC, abstractmethod


class OverloadedException(Exception):
    pass


class Model(ABC):

    @abstractmethod
    def __init__(self, model_name: str, params: dict, system_message: str):
        """ Model implemention should create attributes for all supported parameters """
        self.model_name = model_name
        self.model_params = params # Specific parameters for specific models like temperature
        self.system_message = system_message
        self.debug = params.get('debug', False)

    def set(self, key: str, value):
        if not hasattr(self, key):
            raise (AttributeError(f"Model has no attribute {key}"))
        setattr(self, key, value)

    @abstractmethod
    def chat(self, messages: list[dict], return_json: bool, max_retries: int = 3) -> tuple[[str | object], int, int]:
        pass

    @abstractmethod
    def chat_async(self, messages: list[dict]) -> str:
        pass

    @abstractmethod
    def token_count(self, text: str) -> int:
        pass
