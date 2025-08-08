import os

from dotenv import load_dotenv

from justai import Model


def run_prompt(model_name):
    model = Model(model_name)
    model.system = "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."
    message = model.chat("Forrest Gump", cached=False)
    print(message)
    print(model.last_token_count(), 'tokens')  # (input_token_count, output_token_count, total_token_count)
    print(f'{model.last_response_time:.1f} seconds')


if __name__ == "__main__":
    # cd to the parent directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv()
    for model in ["gpt-5-nano", "claude-3-5-haiku-20241022", "sonar", "deepseek-chat",
                  "openrouter/anthropic/claude-3.5-sonnet-20240620:beta"]:
        print(f"\n******** Running prompt for {model} *************")
        run_prompt(model)
        pass
