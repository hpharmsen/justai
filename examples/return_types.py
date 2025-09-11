"""" Example showing the use of Justai with data returned in a json format """
import pathlib
import json


from justai import Model


def get_story():
    """ Read story.txt from the path of the current file """
    path = pathlib.Path(__file__).parent / 'story.txt'
    with open(path) as f:
        return '<story>\n' + f.read() + '\n</story>\n'


def json_example(model):
    prompt = "Read the following story and give me a list of the persons involved. " + \
             "Return json with keys name, profession and house number\n\n" + get_story()
    return model.prompt(prompt, return_json=True, cached=False)


def structured_output_with_json_schema(model: Model):
    prompt = "Read the following story and give me a list of the persons involved. " + \
             "Return json with keys name, profession and house number\n\n" + get_story()
    schema = {
        "type": "object",
        "properties": {
            "persons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "house_number": {"type": "integer"},
                        "profession": {"type": "string"},
                    },
                    "required": ["name", "house_number", "profession"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["persons"],
        "additionalProperties": False,
    }
    return model.prompt(prompt, return_json=True, response_format=schema, cached=False)


def structured_output_with_pydantic(model: Model):
    """ As of now, only the OpenAI models support structured output with Pydantic"""
    from pydantic import BaseModel
    
    class Person(BaseModel):
        name: str
        house_number: int
        profession: str

    class Persons(BaseModel):
        persons: list[Person]

    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    
    return model.prompt(prompt, response_format=Persons, cached=False)


if __name__ == "__main__":
    data = json_example(Model('claude-sonnet-4-0'))
    print(json.dumps(data, indent=4))

    data = structured_output_with_pydantic(Model('gpt-5'))
    for person in data.persons:
        print(person)
    