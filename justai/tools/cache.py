import hashlib
import os
import json
import sqlite3
from pathlib import Path
from typing import Any

from justdays import Day

from justai.model.message import Message

# Type alias for cache response tuple: (value, tokens_in, tokens_out, extra)
CacheResponse = tuple[str | object, int | None, int | None, dict | None]


def cached_llm_response(model, prompt_or_messages: str | list[Message], tools: list, return_json: bool,
                        response_format=None, use_cache: bool = True, images=None) -> CacheResponse:
    """ Double use function.
    It can be called with
    A: A list of Messages. This is used in chat(). images parameter is ignored.
    B: A string. This is used in prompt(). It uses the images parameter to add images to the prompt
    """

    if not use_cache:
        if isinstance(prompt_or_messages, str):
            return model.prompt(prompt_or_messages, images, tools, return_json, response_format)
        else:
            assert images is None, "When calling cached_llm_response with a string prompt, images should be None"
            return model.chat(prompt_or_messages, tools, return_json, response_format)

    hashcode = recursive_hash(
        (
            model.model_name,
            model.model_params,
            model.system_message,
            prompt_or_messages,
            return_json,
        )
    )

    cachedb = CacheDB()
    result = cachedb.read(hashcode)
    if result:
        if return_json:
            return json.loads(result[0]), result[1], result[2], result[3]
        return result

    if isinstance(prompt_or_messages, str):
        assert hasattr(model, 'prompt')
        result = model.prompt(prompt_or_messages, images, tools, return_json, response_format)
    else:
        assert images is None, "When calling cached_llm_response with a string prompt, images should be None"
        result = model.chat(prompt_or_messages, images, tools, return_json, response_format)
    try:
        if return_json:
            cachedb.write(hashcode, (json.dumps(result[0]), result[1], result[2], result[3]))
        else:
            cachedb.write(hashcode, result)
    except Exception:
        print('cached_llm_response could not write to cache, result is', result)
    return result


def cached_response(*args: Any) -> CacheResponse | None:
    """Retrieve a cached response by hashing the provided arguments."""
    hashcode = recursive_hash((*args,))
    cachedb = CacheDB()
    return cachedb.read(hashcode)


def cache_save(response: tuple[str, int | None, int | None], *args: Any) -> None:
    """Save a response to the cache using hashed arguments as key."""
    hashcode = recursive_hash((*args,))
    cachedb = CacheDB()
    cachedb.write(hashcode, response)


cache_dir = ''
cache_file = 'llmcache.db'


def set_cache_dir(_dir: str | Path) -> None:
    """Set the directory for the cache database."""
    global cache_dir
    cache_dir = str(_dir)


class CacheDB:
    _instance = None

    def __new__(cls, *args, **kwargs):  # Make this class a singleton
        if cls._instance is None:
            cls._instance = super(CacheDB, cls).__new__(cls, *args, **kwargs)
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
        try:
            self.cursor.execute('DELETE FROM cache WHERE valid_until < ?', (str(Day()),))
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def write(self, key: str, llm_response: tuple[str, int | None, int | None], valid_until: str = '') -> None:
        """Write a response to the cache."""
        if not valid_until:
            valid_until = str(Day().plus_months(1))
        value, tokens_in, tokens_out = llm_response  # Ignore tool use
        try:
            self.cursor.execute('''INSERT OR REPLACE INTO cache (hashkey, value, tokens_in, tokens_out, valid_until)
                                    VALUES (?, ?, ?, ?, ?)''', (key, value, tokens_in, tokens_out, valid_until))
            self.conn.commit()
        except sqlite3.ProgrammingError:
            pass  # Something went wrong. Whatever, just don't add to the cache but never crash

    def read(self, key: str) -> tuple[str, int, int, str, str] | None:
        """Read a response from the cache by key."""
        self.cursor.execute("SELECT * FROM cache WHERE hashkey = ?", (key,))
        result = self.cursor.fetchone()
        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            self.cursor.execute('DELETE FROM cache')
            self.conn.commit()
        except sqlite3.ProgrammingError:
            pass  # Something went wrong. Whatever, just don't delete the cache but never crash

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


def recursive_hash(value: Any, depth: int = 0, ignore_params: list[str] | None = None) -> str:
    """Hash primitives recursively with maximum depth."""
    if ignore_params is None:
        ignore_params = []
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


def hash_code(code: str) -> str:
    """Generate MD5 hash of a string."""
    return hashlib.md5(code.encode()).hexdigest()
