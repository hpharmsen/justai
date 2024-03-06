from justai.models.model import Model


class ModelFactory:
    @staticmethod
    def create(model_name: str, **kwargs) -> Model:
        if model_name.startswith("gpt"):
            from justai.models.openai_models import OpenAIModel
            return OpenAIModel(model_name, params=kwargs)
        elif model_name.endswith(".gguf"):
            from justai.models.gguf_models import GuffModel
            return GuffModel(model_name, params=kwargs)
        elif model_name.startswith("claude"):
            from justai.models.anthropic_models import AnthropicModel
            return AnthropicModel(model_name, params=kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported")
