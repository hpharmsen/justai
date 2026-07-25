import base64
import io
import re

import httpx
from PIL import Image

# Standard headers for fetching images from URLs
_HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; JustAI/1.0; +https://github.com/justai)'
}


# --- Robuuste extractie van base64 PNG uit verschillende SDK-vormen ---
def extract_images(response):
    images_b64 = []

    # 1) Nieuwere SDK's plaatsen vaak "message" items met content entries
    for item in getattr(response, "output", []) or []:
        itype = getattr(item, "type", None)

        # a) Message met content -> zoek 'output_image' of 'image'
        if itype == "message":
            for part in getattr(item, "content", []) or []:
                img = getattr(part, "image", None)
                # part.image.base64
                if img is not None and hasattr(img, "base64"):
                    images_b64.append(img.base64)

        # b) Direct image item
        if itype in {"image", "output_image"}:
            img = getattr(item, "image", None)
            if img is not None and hasattr(img, "base64"):
                images_b64.append(img.base64)

        # c) Sommige versies leveren een 'image_generation_call' met 'result'
        # Kan al base64 string zijn
        if itype == "image_generation_call" and hasattr(item, "result") and item.result:
            images_b64.append(item.result)

    # 2) Fallback: oudere voorbeelden met response.output[0].content[0].image.base64
    if not images_b64:
        try:
            maybe = response.output[0].content[0].image.base64
            if maybe:
                images_b64.append(maybe)
        except Exception:
            pass

    if not images_b64:
        # Handige debug: laat zien welke output-types we kregen
        types = [
            getattr(x, "type", type(x).__name__)
            for x in (getattr(response, "output", []) or [])
        ]
        raise RuntimeError(
            f"Geen afbeelding gevonden in response. Output types: {types}"
        )

    return images_b64


def get_image_type(image):
    if isinstance(image, str) and is_image_url(image):
        return 'image_url'
    elif isinstance(image, bytes):
        return 'image_data'
    elif isinstance(image, Image.Image):
        return 'pil_image'
    else:
        raise ValueError("Unknown content type in message. Must be image url or PIL image or image data.")


def _to_bytes(image) -> tuple[bytes, str | None]:
    """Convert image to raw bytes. Returns (data, mime_type) where mime_type is None if unknown."""
    match get_image_type(image):  # Raises ValueError on anything but the three known types
        case 'image_url':
            return httpx.get(image, headers=_HTTP_HEADERS).content, None
        case 'image_data':
            return image, None
        case _:  # pil_image
            buffered = io.BytesIO()
            image.save(buffered, format="jpeg")
            return buffered.getvalue(), 'image/jpeg'


def to_base64_image(image) -> str:
    """Convert image to base64 string."""
    img, _ = _to_bytes(image)
    return base64.b64encode(img).decode("utf-8")


def detect_mime_type(data: bytes) -> str:
    """Detect MIME type from image data using magic bytes."""
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    elif len(data) >= 3 and data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    elif len(data) >= 6 and data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    elif len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    elif len(data) >= 2 and data[:2] == b'BM':
        return 'image/bmp'
    return 'image/jpeg'  # Default fallback


def to_base64_data_uri(image) -> str:
    """Convert image to base64 data URI with proper MIME type detection."""
    img_data, mime_type = _to_bytes(image)
    b64 = base64.b64encode(img_data).decode("utf-8")
    return f"data:{mime_type or detect_mime_type(img_data)};base64,{b64}"


def to_pil_image(image):
    match get_image_type(image):  # Raises ValueError on anything but the three known types
        case 'image_url':
            return Image.open(io.BytesIO(httpx.get(image, headers=_HTTP_HEADERS).content))
        case 'image_data':
            return Image.open(io.BytesIO(image))
        case _:  # pil_image: return as-is, a JPEG roundtrip would lose quality and alpha
            return image


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

    return bool(re.match(url_pattern, url)) and url.lower().endswith(image_extensions)


def crop_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # bron is breder dan doel → crop links/rechts
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        # bron is hoger dan doel → crop boven/onder
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        box = (0, top, src_w, top + new_h)

    return image.crop(box).resize((target_w, target_h), Image.LANCZOS)