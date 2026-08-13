"""Just some color coded output to the terminal"""

import rich

ERROR_COLOR = '#ff0000'
DEBUG_COLOR2 = '#666666'


def color_print(text, color, end='\n'):
    rich.get_console().print(text, style=color, end=end)
