# JustAI

Package to make working with Large Language models in Python super easy.
Supports OpenAI, Anthropic Claude, Google Gemini, X Grok, DeepSeek and open source .guff models.

Author: Hans-Peter Harmsen (hp@harmsen.nl) \
Current version: 4.0.3

Version 4.x is not compatible with the 3.x series.

## Installation
1. Install the package:
~~~~bash
python -m pip install justai
~~~~
2. Create an OpenAI acccount (for OpenAI models) [here](https://platform.openai.com/) or an Anthropic account [here](https://console.anthropic.com/) or a Google account
3. Create an OpenAI api key (for Anthropic) [here](https://platform.openai.com/account/api-keys) or an Anthropic api key [here](https://console.anthropic.com/settings/keys) or a Google api key [here](https://aistudio.google.com/app/apikey)
4. Create a .env file with the following content, depending on the model you intend to use:
```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORGANIZATION=your-openai-organization-id
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
X_API_KEY=your-x-ai-api-key
DEEPSKEEK_API_KEY=your-deepseek-api-key
```
## Usage

```Python
from justai import Model

if __name__ == "__main__":
    model = Model('gpt-4o-mini')
    model.system = "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."

    message = model.chat("Forrest Gump")
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

**OpenAI** models like GPT-4 and O3\
**Anthropic** models like the Claude-3 models\
**Google** models like the Gemini models\
**X AI** models like the Grok models\
**DeekSeek** models like Deepseek V-3 (deepseek-chat) and reasoning model Deepseek-R1 (deepseek-reasoning)\
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
Shows basic use of Justai

### Returning json or other types
```bash
python examples/return_types.py
```
You can specify a specific return type (like a list of dicts) for the completion. 
This is useful when you want to extract structured data from the completion.

To return json, just pass return_json=True to model.chat() and tell the model in the 
prompt how you want your json to be structured.

To define a specific return type you can use the return_type parameter.
Currently this works with the Google models (pass a Python type definition, returns Json)
and with OpenAI (pass a Pydatic type definition, returns a Pydantic model).\
See the example code for more details.

### Images
```bash
python examples/vision.py
```
Pass images to the model. An image can either be:
* An url to an image
* The raw image data
* A PIL image

### Asynchronous use
```bash
python examples/async.py
```
Shows how to use Justai asynchronously.

### Prompt caching
```bash
python examples/prompt_caching.py
```
Shows how to use Prompt caching in Anthropic models.
