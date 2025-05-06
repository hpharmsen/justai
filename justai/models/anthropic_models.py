""" Implementation of the Anthropic models. 

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES
    - Structured types: NO
    - Token count:      YES
    - Image support:    YES
    - Tool use:         not yet
    
Models:
Claude 3 Opus:	 claude-3-opus-20240229
Claude 3 Sonnet: claude-3-5-sonnet-20240620
Claude 3 Haiku:  claude-3-haiku-20240307

Supported parameters:
max_tokens: int (default 800)
temperature: float (default 0.8)

(1) In contrast to Model.chat, Model.chat_async cannot return json and does not return input and output token counts

"""

import json
import os
from typing import Any

import httpx
from anthropic import Anthropic, AsyncAnthropic, APIConnectionError, AuthenticationError, PermissionDeniedError, \
    APITimeoutError, RateLimitError, BadRequestError
from dotenv import dotenv_values
from google.api_core.exceptions import InternalServerError

from justai.model.message import Message
from justai.models.basemodel import BaseModel, identify_image_format_from_base64, ConnectionException, AuthorizationException, \
    ModelOverloadException, RatelimitException, BadRequestException, GeneralException
from justai.tools.display import ERROR_COLOR, color_print


class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, params: dict):
        system_message = f"You are {model_name}, a large language model trained by Anthropic."
        super().__init__(model_name, params, system_message)
        self.cached_prompt = None

        # Authentication
        if "ANTHROPIC_API_KEY" in params:
            api_key = params["ANTHROPIC_API_KEY"]
            del params["ANTHROPIC_API_KEY"]
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY") or dotenv_values()["ANTHROPIC_API_KEY"]
        if not api_key:
            color_print(
                "No Anthropic API key found. Create one at https://console.anthropic.com/settings/keys and "
                + "set it in the .env file like ANTHROPIC_API_KEY=here_comes_your_key.",
                color=ERROR_COLOR,
            )

        # Client
        timeout = httpx.Timeout(30.0)
        if params.get("async"):
            http_client = httpx.AsyncClient(timeout=timeout)
            self.client = AsyncAnthropic(api_key=api_key, http_client=http_client)
        else:
            http_client = httpx.Client(timeout=timeout)
            self.client = Anthropic(api_key=api_key, http_client=http_client)

        # Required model parameters
        if "max_tokens" not in params:
            params["max_tokens"] = 800

    def chat(self, messages: list[Message], tools, return_json: bool, response_format, log=None) \
            -> tuple[str | Any, int, int, dict[str, object | str] | dict[str, Any]]:
        if response_format:
            raise NotImplementedError("Anthropic does not support response_format")

        antr_messages = transform_messages(messages, return_json)
        antr_tools = transform_tools(tools)
        system_message = self.cached_system_message() if self.cached_prompt else self.system_message

        try:
            message = self.client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=antr_messages,
                tools=antr_tools,
                **self.model_params,
            )
        except APIConnectionError as e:
            print("LLM call failed (APIConnectionError):", repr(e))
            raise ConnectionException(e)
        except (AuthenticationError, PermissionDeniedError) as e:
            print("LLM call failed (Auth):", repr(e))
            raise AuthorizationException(e)
        except InternalServerError as e:
            print("LLM call failed (500):", repr(e))
            raise ModelOverloadException(e)
        except RateLimitError as e:
            print("LLM call failed (RateLimit):", repr(e))
            raise RatelimitException(e)
        except BadRequestError as e:
            print("LLM call failed (BadRequest):", repr(e))
            raise BadRequestException(e)
        except Exception as e:
            print("LLM call failed (Unexpected):", repr(e))
            raise GeneralException(e)

        # Text content
        response_str = message.content[0].text
        if return_json:
            response_str = response_str.split("</json>")[0]  # !!
            try:
                response = json.loads(response_str, strict=False)
            except json.decoder.JSONDecodeError:
                print("ERROR DECODING JSON, RESPONSE:", response_str)
                response = response_str
        else:
            response = response_str

        # Tool use
        # if message.stop_reason == "tool_use":
        #     for c in message.content:
        #         if isinstance(c, ToolUseBlock):
        #             tool_use = {'function_to_call': c.name,
        #                         'function_parameters': c.input,
        #                         'call_id': c.id}
        #             break  # For now, only one function call is supported
        # else:
        #     tool_use = {}
        tool_use = {}

        # Token count
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        if self.cached_prompt:
            self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens
            self.cache_read_input_tokens = message.usage.cache_read_input_tokens
        else:
            self.cache_creation_input_tokens = self.cache_read_input_tokens = 0

        return response, input_tokens, output_tokens, tool_use

    def chat_async(self, messages: list[Message]) -> [str, str]:
        try:
            # stream = self.client.messages.create(
            #     model=self.model_name,
            #     max_tokens=self.model_params["max_tokens"],
            #     temperature=self.model_params.get('temperature', 0.8),
            #     system=self.system_message,
            #     messages=transform_messages(messages, return_json=False),
            #     stream=True,
            # )
            stream = self.client.messages.create(
                model=self.model_name,
                system=self.system_message,
                messages=transform_messages(messages, return_json=False),
                stream=True,
                **self.model_params
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

        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                yield event.delta.text, None   # 2nd parameter is reasoning_content. Not available yet for Anthropic

    def cached_system_message(self) -> list[dict]:
        return [
                  {
                    "type": "text",
                    "text": self.system_message,
                  },
                  {
                    "type": "text",
                    "text": self.cached_prompt,
                    "cache_control": {"type": "ephemeral"}
                  }
                ]

    @staticmethod  # !! Veranderen als deze niet anders wordt dan die van OpenAI
    def tool_use_message(tool_use) -> Message:
        return Message(role='user', content='', tool_use=tool_use)

    def token_count(self, text: str) -> int:
        messages = transform_messages([Message("user", text)], return_json=False)
        return self.client.beta.messages.count_tokens(model=self.model_name, messages=messages)


def transform_messages(messages: list[Message], return_json: bool) -> list[dict]:
    # Anthropic requires the first message to be a user message
    msgs = messages[next(i for i, message in enumerate(messages) if message.role == "user"):]

    if msgs and return_json:
        msgs += [Message("assistant", "<json>")]
    result = [create_anthropic_message(msg) for msg in msgs]
    return result


def transform_tools(tools: list[dict]) -> list[dict]:
    """
    At Anthropic tools work like this:
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
    """
    def transform_tool(tool:dict) -> dict:
        if "function" in tool:
            tool["name"] = tool["function"]["name"]
            tool["description"] = tool["function"]["description"]
            tool["input_schema"] = tool["function"]["parameters"]
            del tool["function"]
        if "type" in tool:
            del tool["type"]
        if "function" in tool:
            del tool["function"]
        return tool

    return [transform_tool(tool) for tool in tools]


def create_anthropic_message(message: Message):

    content = []
    role = message.role

    for img in message.images:
        base64img = Message.to_base64_image(img)
        mime_type = identify_image_format_from_base64(base64img)
        content += [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64img,
                },
            }
        ]

    if message.content:
        content += [{"type": "text", "text": message.content}]

    if message.tool_use:
        _input = message.tool_use["function_result"]
        if not isinstance(input, dict): # Anthropic requires input to be a dict
            _input = {'data': _input}
        content += [{
            "type": "tool_use",
            "id": message.tool_use["call_id"],
            "name": message.tool_use["function_to_call"],
            "input": _input,
        }]
        role = 'assistant' # Anthropic requires the role to be assistant when using tools

    return {"role": role, "content": content}
