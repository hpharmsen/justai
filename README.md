# JustAI

Package to make working with Large Language models in Python super easy.
Supports OpenAI, Anthropic Claude, Google Gemini, X Grok, DeepSeek, Perplexity, OpenRouter and open source .guff models.

Author: Hans-Peter Harmsen (hp@harmsen.nl) \
Current version: 4.1.1

Version 4.x is not compatible with the 3.x series.

## Installation
1. Install the package:
~~~~bash
python -m pip install justai
~~~~
2. Create an OpenAI acccount (for OpenAI models) [here](https://platform.openai.com/) or an Anthropic account [here](https://console.anthropic.com/) or a Google account
3. Create an OpenAI api key [here](https://platform.openai.com/account/api-keys) or an Anthropic api key [here](https://console.anthropic.com/settings/keys) or a Google api key [here](https://aistudio.google.com/app/apikey)
4. Create a .env file with the following content, depending on the model you intend to use:
```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORGANIZATION=your-openai-organization-id
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
X_API_KEY=your-x-ai-api-key
DEEPSKEEK_API_KEY=your-deepseek-api-key
```
## Basic usage

```Python
from justai import Model

model = Model('gpt-4o-mini')
model.system = """You are a movie critic. I feed you with movie
                  titles and you give me a review in 50 words."""

message = model.chat("Forrest Gump", cached=True)
print(message)
```
Here, cached=True specifies that justai should cache the prompt and the model's response.

#### output
```
Forrest Gump is an American classic that tells the story of
a man with a kind heart and simple mind who experiences major
events in history. Tom Hanks gives an unforgettable performance, 
making us both laugh and cry. A heartwarming and nostalgic 
movie that still resonates with audiences today.
```
## Models
Justai can use different types of models:

**OpenAI** models like GPT-4 and O3

**Anthropic** models like the Claude-3 models

**Google** models like the Gemini models

**X AI** models like the Grok models

**DeekSeek** models like Deepseek V-3 (deepseek-chat) and reasoning model Deepseek-R1 (deepseek-reasoning)

**Open source** models like Llama2-7b or Mixtral-8x7b-instruct as long as they are in the GGUF format.

**OpenRouter** models. To use these use modelname 'openrouter/_provider_/_modelname'

Except for OpenRouter, the provider is chosen depending on the model name. E.g. if a model name starts with gpt, OpenAI is chosen as the provider.
To use an open source model, just pass the full path to the .gguf file as the model name.


## More advanced usage

### Returning json or other types
```bash
python examples/return_types.py
```
You can specify a specific return type (like a list of dicts) for the completion. 
This is useful when you want to extract structured data from the completion.

To return structured data, just pass return_json=True to model.chat() and tell the model in the 
prompt how you want your json to be structured.

#### Example returning json data
~~~python
model = Model('gemini-1.5-flash')
prompt = "Give me the main characters from Seinfeld with their characteristics. " + \
         "Return json with keys name, profession and weirdness"

data = model.chat(prompt, return_json=True)
print(json.dumps(data, indent=4))
~~~
#### Specifying the return type
To define a specific return type you can use the return_type parameter.

Currently this works with the Google models (pass a Python type definition, returns Json)
and with OpenAI (pass a Pydatic type definition, returns a Pydantic model).


See the example code for more further examples.

### Images
Pass images to the model. An image can either be:
* An url to an image
* The raw image data
* A PIL image

#### Example with PIL image and GPT4o-mini
```python
    
model = Model("gpt-4o-mini-2024-07-18")
url = 'https://upload.wikimedia.org/wikipedia/commons/9/94/Common_dolphin.jpg'
image = Image.open(io.BytesIO(httpx.get(url).content))
message = model.chat("What is in this image", images=url, cached=False)
print(message)

```

### Asynchronous use
```python
async def print_words(model_name, prompt):
    model = Model(model_name)
    async for word in model.chat_async(prompt):
        print(word, end='')
        
prompt = "Give me 5 names for a juice bar that focuses senior citizens."
asyncio.run(print_words("sonar-pro", prompt))
```

### Prompt caching
Shows how to use Prompt caching in Anthropic models.
```python
model = Model('claude-3.7-sonnet')
model.system_message = "You are an experienced book analyzer"  # This is how you set the system message in justai
model.cached_prompt = SOME_STORY
res = model.chat('Who is Mr. Thompsons Neighbour? Give me just the name.',
                 cached=False)  # Disable justai's own cache
print(res)
print('input_token_count', model.input_token_count)
print('output_token_count', model.output_token_count)
print('cache_creation_input_tokens', model.cache_creation_input_tokens)
print('cache_read_input_tokens', model.cache_read_input_tokens)

```

