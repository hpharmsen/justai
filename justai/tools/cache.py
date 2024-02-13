import pickle
from functools import wraps

cache = None
FILE_NAME = "api_cache.pickle"

def cached(func):
    global cache
    if cache is None:
        try:
            with open(FILE_NAME, 'rb') as f:
                cache = pickle.load(f)
        except FileNotFoundError:
            cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__, args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            with open(FILE_NAME, "wb") as fout:
                pickle.dump(cache, fout)
        return cache[key]

    return wrapper
