from justai.models.model import Model


class ModelFactory:
    @staticmethod
    def create(model_name: str, **kwargs) -> Model:
        if model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
            from justai.models.openai_models import OpenAIModel
            return OpenAIModel(model_name, params=kwargs)
        elif model_name.endswith(".gguf"):
            from justai.models.gguf_models import GuffModel
            return GuffModel(model_name, params=kwargs)
        elif model_name.startswith("claude"):
            from justai.models.anthropic_models import AnthropicModel
            return AnthropicModel(model_name, params=kwargs)
        elif model_name.startswith("gemini"):
            from justai.models.google_models import GoogleModel
            return GoogleModel(model_name, params=kwargs)
        elif model_name.startswith("grok"):
            from justai.models.xai_models import XAIModel
            return XAIModel(model_name, params=kwargs)
        elif model_name.startswith("deepseek"):
            from justai.models.deepseek_models import DeepSeekModel
            return DeepSeekModel(model_name, params=kwargs)
        elif model_name.startswith("sonar"):
            from justai.models.perplexity_models import PerplexityModel
            return PerplexityModel(model_name, params=kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported")
