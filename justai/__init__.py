from justai.agent.agent import Agent
from justai.translator.translator import Translator
from justai.interactive.repl import Repl
from justai.interactive.commands import CommandHandler
from justai.tools.prompts import get_prompt, set_prompt_file, add_prompt_file
from justai.tools.log import Log, set_log_dir

if __name__ == '__main__':
    # Ondertaande om de voorkomen dat import optimizer ze leeg gooit
    a = Agent
    t = Translator
    r = Repl
    c = CommandHandler
    g = get_prompt
    s = set_prompt_file
    a = add_prompt_file
    l = Log
    s = set_log_dir
    
