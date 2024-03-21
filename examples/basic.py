from justai import Agent


def run_prompt(model_name):
    agent = Agent(model_name)
    agent.system = "You are a movie critic. I feed you with movie titles and you give me a review in 50 words."
    message = agent.chat("Forrest Gump", cached=True)
    print(message)
    print(agent.last_token_count(), 'tokens')  # (input_token_count, output_token_count, total_token_count)
    print(f'{agent.last_response_time:.1f} seconds')


if __name__ == "__main__":
    for model in ["gpt-3.5-turbo", "gpt-4-turbo-preview", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]:
        print(f"\n******** Running prompt for {model} *************")
        run_prompt(model)
