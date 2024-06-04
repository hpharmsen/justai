import hashlib
import os
import json
import sqlite3
from pathlib import Path

from justdays import Day


def cached_llm_response(model, messages, return_json, use_cache=True, max_retries=None) \
        -> tuple[[str | object], int, int]:
    if not use_cache:
        return model.chat(messages, return_json, max_retries)

    hashcode = recursive_hash((model, messages, return_json))
    cachedb = CachDB()
    result = cachedb.read(hashcode)
    if result:
        if return_json:
            return json.loads(result[0]), result[1], result[2]
        return result
    result = model.chat(messages, return_json, max_retries)
    if return_json:
        cachedb.write(hashcode, (json.dumps(result[0]), result[1], result[2]))
    else:
        cachedb.write(hashcode, result)
    return result


cache_dir = ''
cache_file = 'llmcache.db'


def set_cache_dir(_dir):
    global cache_dir
    cache_dir = _dir


class CachDB:
    _instance = None

    def __new__(cls, *args, **kwargs):  # Make this class a singleton
        if cls._instance is None:
            cls._instance = super(CachDB, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        global cache_dir, cache_file
        dir_ = os.getenv('CACHE_DIR', cache_dir) or Path(__file__).resolve().parent
        self.db_path = os.path.join(dir_, cache_file)
        self.conn = sqlite3.connect(self.db_path)

        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS cache (
                                    hashkey VARCHAR(32) PRIMARY KEY,
                                    value TEXT,
                                    tokens_in INT,
                                    tokens_out INT,
                                    valid_until DATETIME)''')
        self.cursor.execute('DELETE FROM cache WHERE valid_until < ?', (str(Day()),))
        self.conn.commit()

    def write(self, key: str, llm_response: tuple[str, int, int], valid_until: str = ''):
        if not valid_until:
            valid_until = str(Day().plus_months(1))
        value, tokens_in, tokens_out = llm_response
        try:
            self.cursor.execute('''INSERT INTO cache (hashkey, value, tokens_in, tokens_out, valid_until) 
                                    VALUES (?, ?, ?, ?, ?)''', (key, value, tokens_in, tokens_out, valid_until))
            self.conn.commit()
        except sqlite3.ProgrammingError:
            pass # Something went wrong. Whatever, just don't add to the cache but never crash

    def read(self, key):
        self.cursor.execute("SELECT * FROM cache WHERE hashkey = ?", (key,))
        result = self.cursor.fetchone()
        if result:
            return result[1], result[2], result[3]

    def clear(self):
        try:
            self.cursor.execute('DELETE FROM cache')
            self.conn.commit()
        except sqlite3.ProgrammingError:
            pass # Something went wrong. Whatever, just don't delete the cache but never crash

    def close(self):
        self.conn.close()


def recursive_hash(value, depth=0, ignore_params=[]):
    """Hash primitives recursively with maximum depth. Via https://docs.sweep.dev/blogs/file-cache"""
    if depth > 6:
        return hash_code("max_depth_reached")
    if isinstance(value, (int, float, str, bool, bytes)):
        return hash_code(str(value))
    if isinstance(value, (list, tuple)):
        return hash_code("".join([recursive_hash(item, depth + 1, ignore_params) for item in value]))
    if isinstance(value, dict):
        return hash_code(
            "".join(
                [
                    recursive_hash(key, depth + 1, ignore_params)
                    + recursive_hash(val, depth + 1, ignore_params)
                    for key, val in value.items()
                    if key not in ignore_params
                ]
            ))
    if hasattr(value, "__dict__") and value.__class__.__name__ not in ignore_params:
        return recursive_hash(value.__dict__, depth + 1, ignore_params)
    return hash_code("unknown")


def hash_code(code):
    return hashlib.md5(code.encode()).hexdigest()
