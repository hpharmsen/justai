import re

from dotenv import load_dotenv

from justai import Model


def ergobam(x:int, y: int) -> str:
    """ Fictional function that does nothing useful"""
    return 'Twentyseven'


if __name__ == '__main__':
    load_dotenv()
    model = Model("gpt-5-mini")
    model.add_function(ergobam, 'Calculates the ergobam of two numbers',
                       {'x': int, 'y': int}, ['x', 'y'])
    prompt = "Give me the ergobam of 3 and 4."
    response = model.chat(prompt)
    print(response)