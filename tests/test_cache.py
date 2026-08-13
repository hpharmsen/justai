"""Tests for justai.tools.cache — focus on thread-safety of CacheDB.

Usage:
    python tests/test_cache.py
"""

import logging
import os
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import justai.tools.cache as cache_mod
from justai import Model
from justai.model.message import Message
from justai.tools.cache import CacheDB, cached_llm_response, set_cache_dir


def _reset_cachedb_singleton():
    """Forcefully reset the CacheDB singleton between tests."""
    if CacheDB._instance is not None:
        try:
            CacheDB._instance.close()
        except Exception:
            pass
    CacheDB._instance = None


class _MockModel:
    """Minimal stand-in for a justai Model — counts chat() invocations."""

    def __init__(self):
        self.model_name = 'mock-model'
        self.model_params = {'temperature': 0}
        self.system_message = 'you are a test'
        self.call_count = 0
        self._call_lock = threading.Lock()

    def chat(self, messages, images, tools, return_json, response_format):
        with self._call_lock:
            self.call_count += 1
        return ('ok', 1, 2, {})


def test_init_is_idempotent():
    """CacheDB() called repeatedly should not reopen the connection or re-run DDL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        try:
            db1 = CacheDB()
            conn1 = db1.conn
            for _ in range(5):
                db2 = CacheDB()
                assert db2 is db1
                assert db2.conn is conn1
        finally:
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: __init__ idempotent across repeated CacheDB() calls')


def test_parallel_cached_llm_response_single_chat_call():
    """8 threads with identical inputs → 1 underlying model.chat call, no errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        try:
            model = _MockModel()
            messages = [Message(role='user', content='hello there')]

            errors: list[BaseException] = []
            results: list = []
            barrier = threading.Barrier(8)

            def worker():
                try:
                    barrier.wait()
                    res = cached_llm_response(
                        model,
                        messages,
                        tools=[],
                        return_json=False,
                    )
                    results.append(res)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(worker) for _ in range(8)]
                for f in futures:
                    f.result()

            assert not errors, f'workers raised: {errors!r}'
            assert any(isinstance(e, sqlite3.ProgrammingError) for e in errors) is False
            assert len(results) == 8
            for r in results:
                assert r[0] == 'ok'
                assert r[1] == 1
                assert r[2] == 2
            assert model.call_count == 1, f'expected 1 underlying chat call (7 cache hits), got {model.call_count}'
        finally:
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: parallel cached_llm_response → single chat call, no ProgrammingError')


def test_parallel_writes_distinct_keys_no_errors():
    """Concurrent writes with distinct keys should all succeed without sqlite errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        try:
            db = CacheDB()
            errors: list[BaseException] = []

            def writer(i: int):
                try:
                    db.write(f'key_{i}', (f'value_{i}', i, i * 2))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            with ThreadPoolExecutor(max_workers=16) as ex:
                futures = [ex.submit(writer, i) for i in range(32)]
                for f in futures:
                    f.result()

            assert not errors, f'writers raised: {errors!r}'

            for i in range(32):
                row = db.read(f'key_{i}')
                assert row is not None, f'key_{i} missing'
                assert row[0] == f'value_{i}'
                assert row[1] == i
                assert row[2] == i * 2
        finally:
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: parallel writes with distinct keys all persisted')


def test_cache_key_includes_system_prompt():
    """Same prompt with a different system message is a different question."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        try:
            calls = []
            model = Model('gemini-2.5-flash', GEMINI_API_KEY='k')

            def fake_prompt(prompt, images=None, tools=None, return_json=False, response_format=None):
                calls.append(model.model.system_message)
                return ('ok', 1, 2)

            model.model.prompt = fake_prompt

            model.system = 'You are a pirate'
            assert model.prompt('hello') == 'ok'
            assert model.prompt('hello') == 'ok'
            assert len(calls) == 1, 'identical system prompt should hit the cache'

            model.system = 'You are a poet'
            assert model.prompt('hello') == 'ok'
            assert len(calls) == 2, 'different system prompt should miss the cache'
        finally:
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: system prompt is part of the cache key')


def test_json_prompt_hits_the_cache_on_the_second_call():
    """A dict result must survive the cache; without serialisation cached=True was a no-op."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        try:
            calls = []
            model = Model('gemini-2.5-flash', GEMINI_API_KEY='k')
            answer = {'name': 'HP', 'days': [1, 2, 3]}

            def fake_prompt(prompt, images=None, tools=None, return_json=False, response_format=None):
                calls.append(prompt)
                return (answer, 1, 2)

            model.model.prompt = fake_prompt

            assert model.prompt('give me json', return_json=True) == answer
            assert model.prompt('give me json', return_json=True) == answer
            assert len(calls) == 1, 'second JSON call should hit the cache'
        finally:
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: a JSON prompt is served from the cache on the second call')


def test_failed_cache_write_logs_a_warning():
    """A value sqlite cannot bind must not crash, but must not be silent either."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_cachedb_singleton()
        set_cache_dir(tmpdir)
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger('justai.tools.cache')
        logger.addHandler(handler)
        try:
            db = CacheDB()
            db.write('unbindable', ({'a dict': 'sqlite cannot bind'}, 1, 2))
            assert db.read('unbindable') is None
            assert len(records) == 1, f'expected one warning, got {len(records)}'
            assert records[0].levelno == logging.WARNING
            assert 'unbindable' in records[0].getMessage()
        finally:
            logger.removeHandler(handler)
            _reset_cachedb_singleton()
            set_cache_dir('')

    print('  OK: failed cache write logs a WARNING instead of passing silently')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print('Cache tests:')
    test_init_is_idempotent()
    test_parallel_cached_llm_response_single_chat_call()
    test_parallel_writes_distinct_keys_no_errors()
    test_cache_key_includes_system_prompt()
    test_json_prompt_hits_the_cache_on_the_second_call()
    test_failed_cache_write_logs_a_warning()
    print('\nAll cache tests passed!')
