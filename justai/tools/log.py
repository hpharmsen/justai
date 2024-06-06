import json
import os
import sqlite3
from pathlib import Path

from justai.tools.display import color_print

log_dir = ''
log_file = 'justai_log.db'
log_retention_hours = 72

COLORS = {
    "error": "#FF0000",
    "warning": "#EECC00",
    "response": "#0000FF",
    "prompt": "#008000",
    "info": "#888888",
    "default": "#666666"
}


def set_log_dir(_dir):
    global log_dir
    log_dir = _dir


class Log:
    _instance = None

    def __new__(cls, *args, **kwargs):  # Make this class a singleton
        if cls._instance is None:
            cls._instance = super(Log, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        global log_dir, log_file

        dir_ = os.getenv('LOG_DIR', log_dir) or Path(__file__).resolve().parent
        self.log_path = os.path.join(dir_, log_file)
        self.conn = sqlite3.connect(self.log_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS log (
                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                                    title TEXT,
                                    value TEXT,
                                    logtype VARCHAR(10))''')
        self.conn.commit()
        self.cleanup_logs()

    def cleanup_logs(self):
        global log_retention_hours

        sql = "DELETE FROM log WHERE timestamp < datetime('now', ? || ' hours')"
        parameters = (f'-{log_retention_hours}',)
        try:
            # Create a new cursor for the delete operation
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(sql, parameters)
                self.conn.commit()
        except sqlite3.OperationalError as e:
            print(f"OperationalError: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"ProgrammingError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def write(self, title, text, logtype='default'):
        if not isinstance(text, str):
            try:
                text = json.dumps(text, indent=4)
            except Exception:
                text = str(text)
        self.conn.execute('INSERT INTO log (title, value, logtype) VALUES (?, ?, ?)', (title, text, logtype))
        try:
            self.conn.commit()
        except:
            pass # Ignore errors when the log is closed
        print('\n' + title)
        color_print(text, COLORS[logtype])

    def error(self, title, text):
        self.write(title, text, 'error')

    def info(self, title, text):
        self.write(title, text, 'info')

    def warning(self, title, text):
        self.write(title, text, 'warning')

    def prompt(self, title, text):
        self.write(title, text, 'prompt')

    def response(self, title, text):
        self.write(title, text, 'response')

    def clear(self):
        self.cursor.execute('DELETE FROM log')
        self.conn.commit()

    def is_empty(self):
        self.cursor.execute('SELECT COUNT(*) FROM log')
        return self.cursor.fetchone()[0] == 0

    def close(self):
        self.conn.close()

    def as_html(self):
        self.cursor.execute('SELECT * FROM log')
        result = ''
        for res in reversed(self.cursor.fetchall()):
            timestamp, title, value, logtype = res
            value = value.replace('\n', '<br/>')
            color = COLORS.get(logtype, 'gray')
            result += f'<p><b>{timestamp} - {title}</b><br/><span style="color:{color}">{value}</span></p>\n'
        return result


if __name__ == "__main__":
    log = Log()
    log.prompt('prompt', 'System message', )
    print(log.as_html())
    log.response('response', 'This is the response')
    log.error('Oops', 'There was an error')
    log.info('Info', 'This is some information')
    print(log.as_html())