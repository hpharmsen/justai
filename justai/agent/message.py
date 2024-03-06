import json


class Message:
    """ Handles the completion as returned by GPT """
    def __init__(self, role=None, content=None):
        self.role = role
        if not content:
            self.content = None
            self.data = None
        elif isinstance(content, str):
            self.content = content
            self.data = None
        else:
            self.content = json.dumps(content)
            self.data = content

    @classmethod
    def from_dict(cls, data: dict):
        message = cls()
        for key, value in data.items():
            setattr(message, key, value)
        return message

    def __bool__(self):
        return bool(self.content)

    def __str__(self):
        res = f'role: {self.role}'
        res += f' content: {self.content}'
        return res

    def to_dict(self):
        dictionary = {}
        for key, value in self.__dict__.items():
            if value is not None:
                dictionary[key] = value
        return dictionary
