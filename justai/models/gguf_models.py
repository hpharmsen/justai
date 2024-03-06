from llama_cpp import Llama

from justai.models.model import Model


class GuffModel(Model):

    def __init__(self, model_name: str, params: dict):
        """ Model implemention should create attributes for all supported parameters """
        self.model_name = model_name # actually a complete path to the model
        self.system_message = f"You are {model_name.split('/')[-1]}, a large open source language model."

        self.debug = params.get('debug', False)

        # Max tokens for prompt and response combined
        self.n_ctx = params.get('n_ctx', 8192)

        # 0 for CPU only, 1 for Metal, otherwise dependent on GPU-type
        self.n_gpu_layers = params.get('n_gpu_layers', 1)

        self.n_threads = params.get('n_threads', 4)

        self.temperature = params.get('temperature', 0.8)

        self.n_batch = params.get('n_batch', 512)

        self.client = Llama(model_path=model_name, temperature=self.temperature, n_ctx=self.n_ctx,
                            n_batch=self.n_batch, n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers,
                            verbose=False, seed=42)

    def chat(self, prompt: str, return_json: bool, messages: list[dict], use_cache: bool = False,
             max_retries: int = 1) -> tuple[[str | object], int, int]:
        system = messages[0]['content']
        message = f"<s>[INST] <<SYS>>{system}<</SYS>>{messages[-1]['content']}[/INST]"
        output = self.client(message, echo=True, stream=False)
        if self.debug:
            print(output['choices'][0]['text'] + "\n")
        output_text = output['choices'][0]['text'].split('[/INST]')[1].strip()
        if output['choices'][0]['text'][0] == '"' and output_text[-1] == '"':
            output_text = output_text[:-1]
        if not output_text:
            output_text = 'Model produced no result'
        elif output_text[-1] != '.':
            try:
                a, b = output_text.replace('\n', ' ').rsplit('. ', 1)
                output_text = a + '. [' + b + ']'
            except ValueError:
                pass
        prompt_tokens = output['usage']['prompt_tokens']
        completion_tokens = output['usage']['completion_tokens']
        return output_text, prompt_tokens, completion_tokens

    def token_count(self, text: str) -> int:
        return 0

