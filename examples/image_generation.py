import os
import tempfile

from dotenv import load_dotenv
from justai import Model
from PIL.Image import Image


def simple_image_example() -> Image:
    model = Model('reve')
    pil_image = model.generate_image("Create an image of a dolphin reading a book", options={"aspect_ratio": "9:16"})
    return pil_image


def style_transfer_example() -> Image:
    model = Model('gemini-2.5-flash-image-preview')
    url = 'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg'
    pil_image = model.generate_image("Convert this image into the style of van Gogh", images=url)
    return pil_image


def save_and_show(image: Image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        image.save(temp_file.name)
        os.system(f"open {temp_file.name}")


if __name__ == "__main__":
    load_dotenv()

    img1 = simple_image_example()
    save_and_show(img1)

    img2 = style_transfer_example()
    save_and_show(img2)