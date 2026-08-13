import json


class Message:
    """
    Justai uses the Message class to represent messages in a conversation with the model.
    A message has the following attributes:
    - role: The role of the message. Can be "user", "assistant", "system", "function".
    - content: The content of the message. This can be a plainstring or json represented in a string
    - images: A list of images associated with the message.
    """

    def __init__(self, role=None, content=None, images: list = []):
        self.role = role
        if isinstance(content, str):
            self.content = content
        else:
            try:
                self.content = json.dumps(content)
            except (TypeError, OverflowError, ValueError, RecursionError):
                raise ValueError('Invalid content type in message. Must be str or json serializable data.')
        self.images = images

    def __bool__(self):
        return bool(self.content)

    def __str__(self):
        res = f'role: {self.role}'
        res += f' content: {self.content}'
        if self.images:
            res += f' [{len(self.images)} images]'
        return res


class ToolUseMessage(Message):
    def __init__(self, *, content=None, images: list = [], tool_use: dict = None):
        super().__init__('tool', content, images)
        self.tool_use = tool_use
        self.tool_call_id = tool_use.get('call_id', '')
