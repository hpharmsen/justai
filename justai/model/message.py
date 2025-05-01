import base64
import io
import json
import re

import httpx
from PIL import Image


class Message:
    """ Handles the completion as returned by GPT """

    def __init__(self, role=None, content=None, images: list=[], tool_use:dict={}):
        self.role = role
        if isinstance(content, str):
            self.content = content
        else:
            try:
                self.content = json.dumps(content)
            except (TypeError, OverflowError, ValueError, RecursionError):
                raise ValueError("Invalid content type in message. Must be str or json serializable data.")
        self.images = images
        self.tool_use = tool_use

    @classmethod
    def from_dict(cls, data: dict):
        message = cls()
        for key, value in data.items():
            setattr(message, key, value)
        return message

    @staticmethod
    def get_image_type(image):
        if isinstance(image, str) and is_image_url(image):
            return 'image_url'
        elif isinstance(image, bytes):
            return 'image_data'
        elif isinstance(image, Image.Image):
            return 'pil_image'
        else:
            raise ValueError("Unknown content type in message. Must be image url or PIL image or image data.")

    def __bool__(self):
        return bool(self.content)

    def __str__(self):
        res = f'role: {self.role}'
        res += f' content: {self.content}'
        if self.images:
            res += f' [{len(self.images)} images]'
        return res

    def to_dict(self):
        dictionary = {}
        for key, value in self.__dict__.items():
            if value is not None:
                dictionary[key] = value
        return dictionary

    @staticmethod
    def to_base64_image(image):
        image_type = Message.get_image_type(image)
        match image_type:
            case 'image_url':
                img = httpx.get(image).content
            case 'image_data':
                img = image
            case 'pil_image':
                buffered = io.BytesIO()
                image.save(buffered, format="jpeg")
                img = buffered.getvalue()
            case _:
                raise ValueError(f"Unknown image type: {image_type}")
        return base64.b64encode(img).decode("utf-8")

    @staticmethod
    def to_pil_image(image):
        image_type = Message.get_image_type(image)
        match image_type:
            case 'image_url':
                return Image.open(io.BytesIO(httpx.get(image).content))
            case 'image_data':
                return Image.open(io.BytesIO(image))
            case 'pil_image':
                return image
            case _:
                raise ValueError(f"Unknown image type: {image_type}")


def is_image_url(url):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg')
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(url_pattern, url):
        if url.lower().endswith(image_extensions):
            return True
    return False
