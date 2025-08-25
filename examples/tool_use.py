from dotenv import load_dotenv
from justai import Model


def ergobam(x:int, y: int) -> str:
    """ Fictional function that does nothing useful"""
    return 'Twentyseven'


if __name__ == '__main__':
    load_dotenv()
    model = Model("gemini-2.5-flash")
    model.add_tool(ergobam)

    prompt = "Give me the ergobam of 3 and 4."
    response = model.prompt(prompt)
    print(response)