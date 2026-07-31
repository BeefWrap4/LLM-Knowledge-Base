"""Focused tests for configurable local integration endpoints."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from scripts import test_integration


def test_redis_endpoint_uses_environment(monkeypatch):
    captured: dict[str, object] = {}

    class FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.values: dict[str, bytes] = {}

        def ping(self):
            return True

        def info(self, _section):
            return {"redis_version": "test"}

        def set(self, key, value):
            self.values[key] = value.encode()

        def get(self, key):
            return self.values[key]

        def hset(self, key, mapping):
            self.values[key] = str(mapping).encode()

        def keys(self, _pattern):
            return [key for key in self.values if key.startswith("emb:")]

        def delete(self, *keys):
            for key in keys:
                self.values.pop(key, None)

    monkeypatch.setenv("REDIS_HOST", "redis.test")
    monkeypatch.setenv("REDIS_PORT", "26379")
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedis))

    test_integration.test_redis([np.array([1.0, 0.0])])

    assert captured["host"] == "redis.test"
    assert captured["port"] == 26379


def test_pgvector_endpoint_uses_environment(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchone(self):
            return "Python 是一种编程语言", 1.0

    class FakeConnection:
        autocommit = False

        def cursor(self):
            return FakeCursor()

        def close(self):
            return None

    def connect(**kwargs):
        captured.update(kwargs)
        return FakeConnection()

    monkeypatch.setenv("PG_HOST", "postgres.test")
    monkeypatch.setenv("PG_PORT", "25432")
    monkeypatch.setenv("PG_USER", "tutorial")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    monkeypatch.setenv("PG_DATABASE", "vectors")
    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=connect))

    embedding = np.ones(512, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    test_integration.test_pgvector([embedding, embedding, embedding])

    assert captured == {
        "host": "postgres.test",
        "port": 25432,
        "user": "tutorial",
        "password": "secret",
        "dbname": "vectors",
    }
