""" Implementation of the OpenAI Responses APImodels.

Feature table:
    - Async chat:       YES
    - Return JSON:      NO
    - Structured types: YES, via Pydantic  TODO: Add support for native Python types
    - Token count:      YES
    - Image support:    YES
    - Tool use:         YES
    - File input:       NOT YET IMPLEMENTED
    - Web search:       NOT YET IMPLEMENTED
"""
from __future__ import annotations
import base64
import json
import os
from io import BytesIO
from typing import Any, AsyncGenerator
import pydantic
from typing import Any, List, Tuple

from jsonschema import exceptions, validators, Draft202012Validator
import tiktoken
from PIL import Image
from dotenv import dotenv_values
from openai import OpenAI, APIConnectionError, \
    RateLimitError, APITimeoutError, AuthenticationError, PermissionDeniedError, BadRequestError

from justai.model.message import Message, ToolUseMessage
from justai.models.basemodel import ImageInput
from justai.tools.display import color_print, ERROR_COLOR
from justai.models.basemodel import BaseModel, ConnectionException, AuthorizationException, \
    ModelOverloadException, RatelimitException, BadRequestException, GeneralException
from justai.tools.images import extract_images, to_base64_image, get_image_type


class OpenAIResponsesModel(BaseModel):
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

        # Not sure if this works, or is needed, for the Responses API
        # self.client = instructor.patch(OpenAI(api_key=api_key))
        self.client = OpenAI()

        # Diversions from the features that are supported or not supported by default
        self.supports_function_calling = True
        self.supports_image_generation = True

        self.last_response_id = None

    def prompt(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format, _chat=False) \
            -> tuple[Any, int|None, int|None]:

        # if response_format and not issubclass(response_format, pydantic.BaseModel):
        #     raise NotImplementedError("OpenAI Responses API requires response_format to be a Pydantic model.")

        content = self.create_content(prompt, images)
        input_list = [{'role': 'system', 'content': self.system_message},
                      {'role': 'user', 'content': content}]
        tool_spec = self.create_tool_spec(tools)

        last_response_id = self.last_response_id if _chat else None

        for run in range(3):  # Max 3 function calls to prevent infinite loop
            try:
                if response_format and return_json:
                    assert is_valid_json_schema(response_format), "Response format should be a valid JSON Schema"
                    response = self.client.responses.create(model=self.model_name, input=input_list, tools=tool_spec,
                                                            text = {"format": {
                                                                "type": "json_schema",
                                                                "name": "response_format",
                                                                "strict": True,
                                                                "schema":response_format}},
                                                            previous_response_id=last_response_id)
                elif response_format:
                    assert issubclass(response_format, pydantic.BaseModel), 'Response format should be a Pydantic model unless you specify return_json=True'
                    response = self.client.responses.parse(model=self.model_name, input=input_list, tools=tool_spec,
                                                           text_format=response_format,
                                                           previous_response_id=last_response_id)
                elif return_json:
                    response = self.client.responses.create(model=self.model_name, input=input_list, tools=tool_spec,
                                                            text = { "format": { "type": "json_object" } },
                                                            previous_response_id=last_response_id)
                else:
                    response = self.client.responses.create(model=self.model_name, input=input_list, tools=tool_spec,
                                                            previous_response_id=last_response_id)
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

            # Save the response id for subsequent requests
            self.last_response_id = response.id if _chat else None

            # Save function call outputs for subsequent requests
            function_call = None
            function_call_arguments = None
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    function_call = item
                    function_call_arguments = json.loads(item.arguments)

            if not function_call or run == 2:
                if issubclass(response_format, pydantic.BaseModel):
                    field = list(response.output_parsed.model_fields_set)[0]
                    output = getattr(response.output_parsed, field)
                elif return_json:
                    output = json.loads(response.output_text)
                else:
                    output = response.output_text
                return output, response.usage.input_tokens, response.usage.output_tokens

            # The rest of the function is to process the function call output
            # Numbering is from OpenAI's docs: https://platform.openai.com/docs/guides/function-calling#function-tool-example

            for tool in tools:
                if tool['function'].__name__ == function_call.name:
                    function = tool['function']
                    break
            else:
                raise ValueError(f"Function {function_call.name} not found")

            # 3. Execute the function logic for get_horoscope
            result = {function_call.name: function(*function_call_arguments.values())}

            # 4. Provide function call results to the model
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result),
                }
            )
        # Response looks like this:
        # [{
        #   "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
        #   "object": "response",
        #   "created_at": 1741476542,
        #   "status": "completed",
        #   "error": null,
        #   "incomplete_details": null,
        #   "instructions": null,
        #   "max_output_tokens": null,
        #   "model": "gpt-4.1-2025-04-14",
        #   "output": [
        #     {
        #       "type": "message",
        #       "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
        #       "status": "completed",
        #       "role": "assistant",
        #       "content": [
        #         {
        #           "type": "output_text",
        #           "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
        #           "annotations": []
        #         }
        #       ]
        #     }
        #   ],
        #   "parallel_tool_calls": true,
        #   "previous_response_id": null,
        #   "reasoning": {
        #     "effort": null,
        #     "summary": null
        #   },
        #   "store": true,
        #   "temperature": 1.0,
        #   "text": {
        #     "format": {
        #       "type": "text"
        #     }
        #   },
        #   "tool_choice": "auto",
        #   "tools": [],
        #   "top_p": 1.0,
        #   "truncation": "disabled",
        #   "usage": {
        #     "input_tokens": 36,
        #     "input_tokens_details": {
        #       "cached_tokens": 0
        #     },
        #     "output_tokens": 87,
        #     "output_tokens_details": {
        #       "reasoning_tokens": 0
        #     },
        #     "total_tokens": 123
        #   },
        #   "user": null,
        #   "metadata": {}
        # }]


    async def prompt_async(self, prompt: str, images: list[ImageInput]=None, _chat=False) -> AsyncGenerator[str]:

        content = self.create_content(prompt, images)
        input_ = [{"role": "user", "content": content}]

        last_response_id = self.last_response_id if _chat else None

        response = self.client.responses.create(model=self.model_name, input=input_, stream=True,
                                                previous_response_id=last_response_id)

        # Save the response id for subsequent requests
        self.last_response_id = response.id if _chat else None

        for event in response:
            if hasattr(event, 'delta'):
                yield event.delta
        # Events can have different types:
        # event: response.created
        # data: {
        #     "type":"response.created",
        #     "response":{
        #         "id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
        #         "object":"response",
        #         "created_at":1741290958,
        #         "status":"in_progress",
        #         "error":null,
        #         "incomplete_details":null,
        #         "instructions":"You are a helpful assistant.",
        #         "max_output_tokens":null,
        #         "model":"gpt-4.1-2025-04-14","
        #         output":[],"p
        #         arallel_tool_calls":true,
        #         "previous_response_id":null,
        #         "reasoning":{
        #             "effort":null,
        #             "summary":null},
        #         "store":true,
        #         "temperature":1.0,
        #         "text":{
        #             "format":{
        #                 "type":"text"}},
        #         "tool_choice":"auto",
        #         "tools":[],
        #         "top_p":1.0,
        #         "truncation":
        #         "disabled",
        #         "usage":null,
        #         "user":null,
        #         "metadata":{}}}
        #
        # event: response.in_progress
        # data: {
        #     "type":"response.in_progress",
        #     "response":{"
        #         id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
        #         "object":"response",
        #         "created_at":1741290958,
        #         "status":"in_progress",
        #         "error":null,
        #         "incomplete_details":null,
        #         "instructions":"You are a helpful assistant.",
        #         "max_output_tokens":null,
        #         "model":"gpt-4.1-2025-04-14",
        #         "output":[],
        #         "parallel_tool_calls":true,
        #         "previous_response_id":null,
        #         "reasoning":{
        #             "effort":null,
        #             "summary":null},
        #         "store":true,
        #         "temperature":1.0,
        #         "text":{
        #             "format":{
        #                 "type":"text"}},
        #         "tool_choice":"auto",
        #         "tools":[],
        #         "top_p":1.0,
        #         "truncation":"disabled",
        #         "usage":null,
        #         "user":null,
        #         "metadata":{}}}
        #
        # event: response.output_item.added
        # data: {
        #     "type":"response.output_item.added",
        #     "output_index":0,
        #     "item":{
        #         "id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
        #         "type":"message","status":"
        #         in_progress","
        #         role":"assistant",
        #         "content":[]}}
        #
        # event: response.content_part.added
        # data: {
        #     "type":"response.content_part.added",
        #     "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
        #     "output_index":0,
        #     "content_index":0,
        #     "part":{
        #         "type":"output_text",
        #         "text":"",
        #         "annotations":[]}}
        #
        # event: response.output_text.delta
        # data: {
        #     "type":"response.output_text.delta",
        #     "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
        #     "output_index":0,
        #     "content_index":0,
        #     "delta":"Hi"}
        #
        # ...
        #
        # event: response.output_text.done
        # data: {"type":"response.output_text.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"text":"Hi there! How can I assist you today?"}
        #
        # event: response.content_part.done
        # data: {"type":"response.content_part.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}}
        #
        # event: response.output_item.done
        # data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}}
        #
        # event: response.completed
        # data: {"type":"response.completed","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"completed","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":{"input_tokens":37,"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48},"user":null,"metadata":{}}}

    def chat(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format) \
             -> tuple[Any, int|None, int|None]:
        return self.prompt(prompt, images, tools, return_json, response_format, _chat=True)


    async def chat_async(self, prompt: str, images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        async for chunk in self.prompt_async(prompt, images, _chat=True):
            yield chunk
               
    @staticmethod
    def create_content(prompt, images:list[ImageInput]) -> list[dict]:
        content = [{"type": "input_text", "text": prompt}]

        if images:
            for image in images:
                image_type = get_image_type(image)
                image_url = image if image_type == "image_url" else to_base64_image(image)
                content += [{"type": "input_image", "image_url": image_url}]
        return content

    @staticmethod
    def create_tool_spec(tools: list[dict]) -> list[dict]:
        if not tools:
            return []

        type_mapping = {
            int: "integer",
            str: "string",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        tool_spec = []
        for tool in tools:
            tool_spec += [{
                "type": "function",
                "name": tool["function"].__name__,
                "description": tool['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        param_name: {
                            "type": type_mapping.get(_type, "string"),  # Default to string if type not in map
                            "description": param_name,  # Or a more descriptive text if available
                        }
                        for param_name, _type in tool['parameters'].items()
                    },
                    "required": tool['required_parameters'] or [],
                }
            }]
        return tool_spec

    @staticmethod
    def transform_messages(messages: list[Message]) -> list[dict]:
        transformed_messages = []

        for message in messages:
            msg = {"role": message.role}

            # Handle tool messages (function calls and their results)
            # Todo: tool use is geen onderdeel meer van message maar staat in de Model class.
            if isinstance(message, ToolUseMessage):
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
                            "image_url": {'url': f"data:image/jpeg;base64,{to_base64_image(image)}"}
                        })
                    msg["content"] = content
                else:
                    msg["content"] = message.content or ""

            transformed_messages.append(msg)

        return transformed_messages

    def token_count(self, text: str) -> int:
        """ Returns the number of tokens in a string. """
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fall back to cl100k_base encoding for newer models not yet in tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


    def generate_image(self, prompt, images: ImageInput):

        # Responses API call met tool "image_generation"
        client = OpenAI()
        input_structure = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ],
            }
        ]
        if images:
            for image in images:
                if get_image_type(image) == "image_url":
                    image_dict = {"type": "input_image", "image_url": image}
                else:
                    image_dict = {"type": "input_image", "image_data": to_base64_image(image)}
                input_structure[0]["content"] += [image_dict]

        resp = client.responses.create(
            model=self.model_name,
            input=input_structure,
            tools=[{"type": "image_generation"}],
        )

        images_b64 = extract_images(resp)

        # Decode base64 -> PIL image
        raw = base64.b64decode(images_b64[0])
        img = Image.open(BytesIO(raw))
        return img


def is_valid_json_schema(schema: Any) -> Tuple[bool, List[str]]:
    """
    Check whether a Python object is a *valid JSON Schema* (meta-schema validation).

    It auto-detects the draft via `$schema` when present and falls back to Draft 2020-12.
    Returns (is_valid, errors) where errors is a list of human-readable messages.

    Parameters
    ----------
    schema : Any
        The candidate JSON Schema as a Python dict (or JSON-loaded structure).

    Returns
    -------
    Tuple[bool, List[str]]
        - True and [] when the input is a valid JSON Schema.
        - False and a list of error messages when invalid.

    Notes
    -----
    - This validates the *schema itself* against the appropriate meta-schema.
    - It does not validate an instance/document *against* the schema.
    """

    def _format_iter_error(err: exceptions.ValidationError) -> str:
        """Human-friendly error message for iter_errors results."""
        # Build a JSON Pointer–like path for clarity
        path = "/" + "/".join(map(str, err.path)) if err.path else "(root)"
        schema_path = "#/" + "/".join(map(str, err.schema_path)) if err.schema_path else "#"
        return f"[at {path}] {err.message}  (schema path: {schema_path})"


    def _format_schema_error(err: exceptions.SchemaError) -> str:
        """Format SchemaError raised by check_schema()."""
        path = "/" + "/".join(map(str, err.path)) if getattr(err, "path", None) else "(root)"
        schema_path = "#/" + "/".join(map(str, err.schema_path)) if getattr(err, "schema_path", None) else "#"
        base = f"[at {path}] {err.message}  (schema path: {schema_path})"
        if err.context:
            # Include nested context messages (useful for 'oneOf', 'anyOf' diagnostics)
            ctx = "; ".join(_format_iter_error(c) for c in err.context)  # type: ignore[arg-type]
            return f"{base} | context: {ctx}"
        return base

    error_messages: List[str] = []

    # 1) Pick the best validator class for the provided schema (uses `$schema` if present)
    try:
        ValidatorClass = validators.validator_for(schema, default=Draft202012Validator)  # type: ignore[attr-defined]
    except Exception as e:
        return False, [f"Unable to choose validator for schema (is it a dict?): {e!r}"]

    # 2) Fast sanity check (raises on the first structural problem)
    try:
        ValidatorClass.check_schema(schema)
    except exceptions.SchemaError as e:
        # We still continue to collect *all* errors below for a complete report
        error_messages.append(_format_schema_error(e))

    # 3) Full meta-schema validation to gather all issues
    try:
        meta_validator = ValidatorClass(ValidatorClass.META_SCHEMA)  # type: ignore[arg-type]
        errors = sorted(meta_validator.iter_errors(schema), key=lambda err: (list(err.path), err.message))
        if errors:
            for err in errors:
                error_messages.append(_format_iter_error(err))
    except Exception as e:
        # If meta-validation itself fails, report that clearly
        error_messages.append(f"Meta-validation failed: {e!r}")

    return (len(error_messages) == 0), error_messages
