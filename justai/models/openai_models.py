""" Implementation of the OpenAI models. 

Feature table:
    - Async chat:       YES
    - Return JSON:      YES
    - Structured types: YES, via Pydantic  TODO: Add support for native Python types
    - Token count:      YES
    - Image support:    YES
    - Tool use:         not yet

Supported parameters:    
    `# The maximum number of tokens to generate in the completion.
    # Defaults to 16
    # The token count of your prompt plus max_tokens cannot exceed the model's context length.
    # Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
    self.model_params['max_tokens'] = params.get('max_tokens', 800)

    # What sampling temperature to use, between 0 and 2.
    # Higher values like 0.8 will make the output more random, while lower values like 0.2
    # will make it more focused and deterministic.
    # We generally recommend altering this or top_p but not both
    # Defaults to 1
    self.model_params['temperature'] = params.get('temperature', 0.5)

    # An alternative to sampling with temperature, called nucleus sampling,
    # where the model considers the results of the tokens with top_p probability mass.
    # So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    # We generally recommend altering this or temperature but not both.
    # Defaults to 1
    self.model_params['top_p'] = params.get('top_p', 1)

    # How many completions to generate for each prompt.
    # Because this parameter generates many completions, it can quickly consume your token quota.
    # Use carefully and ensure that you have reasonable settings for max_tokens.
    self.model_params['n'] = params.get('n', 1)

    # Number between -2.0 and 2.0.
    # Positive values penalize new tokens based on whether they appear in the text so far,
    # increasing the model's likelihood to talk about new topics.
    # Defaults to 0
    self.model_params['presence_penalty'] = params.get('presence_penalty', 0)

    # Number between -2.0 and 2.0.
    # Positive values penalize new tokens based on their existing frequency in the text so far,
    # decreasing the model's likelihood to repeat the same line verbatim.
    # Defaults to 0
    self.model_params['frequency_penalty'] = params.get('frequency_penalty', 0)
"""

import json
import os

import tiktoken
from dotenv import dotenv_values
from openai import OpenAI, NOT_GIVEN, APIConnectionError, \
    RateLimitError, APITimeoutError, AuthenticationError, PermissionDeniedError, BadRequestError

from justai.model.message import Message
from justai.tools.display import color_print, ERROR_COLOR, DEBUG_COLOR1, DEBUG_COLOR2
from justai.models.basemodel import BaseModel, ConnectionException, AuthorizationException, \
    ModelOverloadException, RatelimitException, BadRequestException, GeneralException


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by OpenAI."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = params.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or dotenv_values()["OPENAI_API_KEY"]
        if not api_key:
            color_print("No OpenAI API key found. Create one at https://platform.openai.com/account/api-keys and " +
                        "set it in the .env file like OPENAI_API_KEY=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages: list[Message], tools: list, return_json: bool, response_format, use_cache: bool = False) -> tuple[str, int, int, dict|None]:
        # OpenAI models like to have  the system message as part of the conversation
        messages = [Message('system', self.system_message)] + messages

        if self.debug:
            color_print("\nRunning completion with these messages", color=DEBUG_COLOR1)
            [color_print(m, color=DEBUG_COLOR1) for m in messages if hasattr(m, 'text')]
            print()

        try:
            completion = self.completion(messages, tools, return_json, response_format)
        except APIConnectionError as e:
            raise ConnectionException(e)
        except APITimeoutError as e:
            raise ModelOverloadException(e)
        except (AuthenticationError, PermissionDeniedError) as e:
            raise AuthorizationException(e)
        except RateLimitError as e:
            raise RatelimitException(e)
        except BadRequestError as e:
            raise BadRequestException(e)
        except Exception as e:
            raise GeneralException(e)

        message = completion.choices[0].message
        message_text = message.content

        # Tool use
        # if completion.choices[0].finish_reason == 'tool_calls':
        #     f = completion.choices[0].message.tool_calls[0].function
        #     tool_use = {
        #         "function_to_call": f.name,
        #         "function_parameters": json.loads(f.arguments),
        #         "call_id": completion.id,
        #     }
        # else:
        #     tool_use = {}
        tool_use = {}

        # Token counts
        input_token_count = completion.usage.prompt_tokens
        output_token_count = completion.usage.completion_tokens

        if message_text and message_text.startswith('```json'):
            print('Unexpected JSON response found in OpenAI completion')
            message_text = message_text[7:-3]
        if self.debug:
            color_print(f"{message_text}", color=DEBUG_COLOR2)

        if response_format and completion.choices[0].message.parsed:
            result = completion.choices[0].message.parsed
        else:
            result = json.loads(message_text) if return_json else message_text
        return result, input_token_count, output_token_count, tool_use
    
    def chat_async(self, messages: list[Message]):
        try:
            completion = self.completion(messages, stream=True)
        except APIConnectionError as e:
            raise ConnectionException(e)
        except (AuthenticationError, PermissionDeniedError) as e:
            raise AuthorizationException(e)
        except APITimeoutError as e:
            raise ModelOverloadException(e)
        except RateLimitError as e:
            raise RatelimitException(e)
        except BadRequestError as e:
            raise BadRequestException(e)
        except Exception as e:
            raise GeneralException(e)

        for item in completion:
            content = item.choices[0].delta.content if hasattr(item.choices[0].delta, "content") else None
            reasoning = item.choices[0].delta.reasoning_content if hasattr(item.choices[0].delta, "reasoning_content") else None
            if content or reasoning:
                yield content, reasoning
               
    def completion(self, messages: list[Message], tools, return_json: bool = False, response_format: BaseModel = None,
                   stream: bool = False):
        transformed_messages = self.transform_messages(messages)
        
        if response_format:
            if stream:
                raise NotImplementedError("streaming is not supported with response_format")
            if tools:
                raise NotImplementedError("tools is not supported with response_format")
            return self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=transformed_messages,
                response_format=response_format,
                **self.model_params
            )
        else:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=transformed_messages,
                #tools=tools,
                response_format={"type": "json_object"} if return_json else NOT_GIVEN,
                stream=stream,
                **self.model_params
            )
    
    @staticmethod
    def transform_messages(messages: list[Message]) -> list[dict]:
        def create_openai_message(message):
            msg = {"role": message.role}
            if message.tool_use:
                msg['name'] = message.tool_use['function_to_call']
                msg['content'] = message.tool_use['function_result']
            else:
                content = [{"type": "text", "text": message.content}]
                for image in message.images:
                    content += [{
                                    "type": "image_url",
                                    "image_url": {'url': f"data:image/jpeg;base64,{Message.to_base64_image(image)}"}
                               }]
                msg["content"] = content
            return msg

        result = [create_openai_message(msg) for msg in messages]
        return result

    @staticmethod
    def tool_use_message(tool_use) -> Message:
        return Message(role='function', content='', tool_use=tool_use)

    def token_count(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))
