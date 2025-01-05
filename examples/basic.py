import os

from dotenv import load_dotenv

from justai import Agent


def run_prompt(model_name):
    agent = Agent(model_name)
    agent.system = "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."
    message = agent.chat("Forrest Gump", cached=False)
    print(message)
    print(agent.last_token_count(), 'tokens')  # (input_token_count, output_token_count, total_token_count)
    print(f'{agent.last_response_time:.1f} seconds')


if __name__ == "__main__":
    # cd to the parent directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv()
    for model in ["deepseek-chat", "claude-3-5-haiku-20241022", "gpt-4o-mini", "gemini-1.5-flash"]:
        print(f"\n******** Running prompt for {model} *************")
        run_prompt(model)
        pass
