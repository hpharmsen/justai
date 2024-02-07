import json


class Message:
    """ Handles the completion as returned by GPT """
    def __init__(self, role=None, text_or_json=None, function_name=None, function_content=None):
        self.role = role
        if not text_or_json:
            self.text = None
            self.data = None
        elif isinstance(text_or_json, str):
            self.text = text_or_json
            self.data = None
        else:
            self.text = json.dumps(text_or_json)
            self.data = text_or_json

    @classmethod
    def from_dict(cls, data: dict):
        message = cls()
        for key, value in data.items():
            setattr(message, key, value)
        return message

    def __bool__(self):
        return bool(self.text)

    def __str__(self):
        res = f'role: {self.role}'
        res += f' text: {self.text}'
        return res

    def to_dict(self):
        dictionary = {}
        for key, value in self.__dict__.items():
            if value is not None:
                dictionary[key] = value
        return dictionary
