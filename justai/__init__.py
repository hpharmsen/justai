from justai.model.model import Model
from justai.tools.prompts import get_prompt, set_prompt_file, add_prompt_file

if __name__ == '__main__':
    # Ondertaande om de voorkomen dat import optimizer ze leeg gooit
    a = Model
    g = get_prompt
    s = set_prompt_file
    apf = add_prompt_file

