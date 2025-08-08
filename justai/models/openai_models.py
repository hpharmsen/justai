""" Implementation of the OpenAI models. 

Feature table:
    - Async chat:       YES
    - Return JSON:      YES
    - Structured types: YES, via Pydantic  TODO: Add support for native Python types
    - Token count:      YES
    - Image support:    YES
    - Tool use:         YES

Supported parameters:    
    # The maximum number of tokens to generate in the completion.
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
from typing import Any

import instructor
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

        # instructor.patch makes the OpenAI client compatible with structured output via response_model=”
        # Works only for OpenAI models
        self.client = instructor.patch(OpenAI(api_key=api_key))

    def chat(self, messages: list[Message], tools: list, return_json: bool, response_format, use_cache: bool = False) \
            -> tuple[Any, int|None, int|None, dict|None]:

        if tools and response_format:
            raise NotImplementedError("OpenAI API does not support both tools and response_format")

        # OpenAI models like to have the system message as part of the conversation
        messages = [Message('system', self.system_message)] + messages

        if self.debug:
            color_print("\nRunning completion with these messages", color=DEBUG_COLOR1)
            [color_print(m, color=DEBUG_COLOR1) for m in messages if hasattr(m, 'text')]
            print()

        if not tools: # Models like deepseek-chat don't like tools to be an empty list
            tools = NOT_GIVEN

        try:
            completion = self.completion(messages, tools, return_json, response_model=response_format)
        except APITimeoutError as e:
            raise ModelOverloadException(e)
        except APIConnectionError as e:
            raise ConnectionException(e)
        except (AuthenticationError, PermissionDeniedError) as e:
            raise AuthorizationException(e)
        except RateLimitError as e:
            raise RatelimitException(e)
        except BadRequestError as e:
            raise BadRequestException(e)
        except Exception as e:
            raise GeneralException(e)

        if response_format:
            # Intended behavior bij OpenAI. When response_format is specified, the raw response is alreay
            # deserialized into the requested format.
            # Disadvantage: the raw response is not available so no token count or tool use
            return completion, None, None, None

        message = completion.choices[0].message
        message_text = message.content
        input_token_count = completion.usage.prompt_tokens
        output_token_count = completion.usage.completion_tokens
        result = json.loads(message_text) if return_json and self.supports_return_json else message_text

        # Tool use
        if message.tool_calls and not response_format:
            f = message.tool_calls[0].function
            tool_use = {
                "function_to_call": f.name,
                "function_parameters": json.loads(f.arguments),
                "call_id": completion.id if not response_format else raw_response.id,
            }
        else:
            tool_use = {}

        if message_text and message_text.startswith('```json'):
            print('Unexpected JSON response found in OpenAI completion')
            message_text = message_text[7:-3]
        if self.debug:
            color_print(f"{message_text}", color=DEBUG_COLOR2)

        return result, input_token_count, output_token_count, tool_use
    
    def chat_async(self, messages: list[Message]):
        try:
            # Pass required parameters to completion method
            completion = self.completion(
                messages=messages,
                tools=None,  # No tools for async streaming
                return_json=False,  # Default to False for streaming
                response_model=None,  # No response format for streaming
                stream=True
            )
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
               
    def completion(self, messages: list[Message], tools=NOT_GIVEN, return_json: bool = False,
                   response_model: BaseModel = None, stream: bool = False):
        transformed_messages = self.transform_messages(messages)

        if response_model:
            if not "openai.com" in str(self.client.base_url):
                raise NotImplementedError("response_model is only supported with OpenAI models")
            if stream:
                raise NotImplementedError("streaming is not supported with response_model")
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=transformed_messages,
                response_model=response_model,
                **self.model_params
            )

        if return_json and not stream and self.supports_return_json:
            self.model_params['response_format'] = {"type": "json_object"}

        if self.model_name.startswith("gpt-5"):
            self.model_params["temperature"] = 1  # Only the default of 1 is supported in GPT-5

        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=transformed_messages,
            tools=tools,
            stream=stream,
            **self.model_params
        )

        if 'response_format' in self.model_params:
            del self.model_params['response_format']

        return result

    @staticmethod
    def transform_messages(messages: list[Message]) -> list[dict]:
        transformed_messages = []

        for message in messages:
            msg = {"role": message.role}

            # Handle tool messages (function calls and their results)
            if message.tool_use:
                if message.role == 'assistant' and 'function_to_call' in message.tool_use:
                    # This is a function call from the assistant
                    msg["content"] = None
                    msg["tool_calls"] = [{
                        "id": message.tool_use.get('call_id', 'call_' + str(hash(str(message.tool_use)))),
                        "type": "function",
                        "function": {
                            "name": message.tool_use['function_to_call'],
                            "arguments": json.dumps(message.tool_use['function_parameters'])
                        }
                    }]
                elif message.role == 'tool':
                    # This is a function result
                    function_result = message.tool_use.get('function_result', '')
                    if not isinstance(function_result, str):
                        function_result = json.dumps(function_result)
                    msg["content"] = function_result
                    msg["tool_call_id"] = message.tool_use.get('call_id', '')
                    msg["name"] = message.tool_use.get('function_to_call', '')
            # Handle regular messages
            else:
                if message.images:
                    content = [{"type": "text", "text": message.content or ""}]
                    for image in message.images:
                        content.append({
                            "type": "image_url",
                            "image_url": {'url': f"data:image/jpeg;base64,{Message.to_base64_image(image)}"}
                        })
                    msg["content"] = content
                else:
                    msg["content"] = message.content or ""

            transformed_messages.append(msg)

        return transformed_messages

    @staticmethod
    def tool_use_message(tool_use) -> Message:
        """ Creates a message with the result of a tool use. """
        return Message('tool', tool_use=tool_use)

    def token_count(self, text: str) -> int:
        """ Returns the number of tokens in a string. """
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))
