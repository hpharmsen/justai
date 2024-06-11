# JustAI

Package to make working with Large Language models in Python super easy.

Author: Hans-Peter Harmsen (hp@harmsen.nl) \
Current version: 3.4.5

## Installation
1. Install the package:
~~~~bash
python -m pip install justai
~~~~
2. Create an OpenAI acccount (for GPT3.5 / 4) [here](https://platform.openai.com/) or an Anthropic account [here](https://console.anthropic.com/)
3. Create an OpenAI api key (for Claude) [here](https://platform.openai.com/account/api-keys) or an Anthropic api key [here](https://console.anthropic.com/settings/keys)
4. Create a .env file with the following content:
```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORGANIZATION=your-openai-organization-id
ANTHROPIC_API_KEY=your-anthropic-api-key
```
## Usage

```Python
from justai import Agent

if __name__ == "__main__":
    agent = Agent('gpt-3.5-turbo')
    agent.system = "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."

    message = agent.chat("Forrest Gump")
    print(message)
```
output
```
Forrest Gump is an American classic that tells the story of
a man with a kind heart and simple mind who experiences major
events in history. Tom Hanks gives an unforgettable performance, 
making us both laugh and cry. A heartwarming and nostalgic 
movie that still resonates with audiences today.
```
## Other models
Justai can use different types of models:

**OpenAI** models like GPT-3.5, GPT-4-turbo-preview\
**Anthropic** models like claude-3-opus-20240229 and claude-3-sonnet-20240229\
**Open source** models like Llama2-7b or Mixtral-8x7b-instruct as long as they are in the GGUF format.

The provider is chosen depending on the model name. E.g. if a model name starts with gpt, OpenAI is chosen as the provider.
To use an open source model, just pass the full path to the .gguf file as the model name.


## Using the examples
Install dependencies:
```bash
python -m pip install -r requirements.txt
```


### Basic
```bash
python examples/basic.py
```
Starts an interactive session. In the session you dan chat with GPT-4 or another model.

### Returning json
```bash
python examples/return_types.py
```
You can specify a specific return type (like a list of dicts) for the completion. 
This is useful when you want to extract structured data from the completion.

To define a return type, just pass return_json=True to agent.chat().

See the example code for more details.

### Interactive
```bash
python examples/interactive.py
```
Starts an interactive session. In the session you dan chat with GPT-4 or another model.

#### Special commands
In the interactive mode you can use these special commands which each start with a colon:

| Syntax                            | Description                                                         |
|-----------------------------------|---------------------------------------------------------------------|
| :reset                            | resets the conversation                                             |
| :load _name_                      | loads the saved conversation with the specified name                |
| :save _name_                      | saves the conversation under the specified name                     |
| :input _filename_                 | loads an input from the specified file                              |
| :model _gpt-4_                    | Sets the AI model                                                   |
| :max_tokens _800_                 | The maximum number of tokens to generate in the completion          |
| :temperature _0.9_                | What sampling temperature to use, between 0 and 2                   |
| :n _1_                            | Specifies the number answers given                                  |
| :stop _["\n", " Human:", " AI:"]_ | Up to 4 sequences where the API will stop generating further tokens |
| :bye                              | quits but saves the conversation first                              |
| :exit or :quit                    | quits the program                                                   |

