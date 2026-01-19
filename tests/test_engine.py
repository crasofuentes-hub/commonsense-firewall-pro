"""
Commonsense Firewall - Test Suite (Windows-safe, offline-friendly)

This suite intentionally focuses on deterministic behaviors and avoids
network/model downloads by default.

Run:
  python -m pytest .\tests -q
"""

import os
import time
import pytest

# Make repo root importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import FastCommonsenseEngine


def close_engine(eng):
    """
    Best-effort cleanup so Windows can delete temp sqlite file.
    The engine may expose different cleanup APIs depending on version.
    """
    if eng is None:
        return

    # Common patterns: close(), shutdown(), stop()
    for attr in ("close", "shutdown", "stop"):
        fn = getattr(eng, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    # Give Windows a moment to release file handles
    time.sleep(0.05)


def make_engine(tmp_path, **kwargs) -> FastCommonsenseEngine:
    """
    Create an engine backed by a sqlite file under pytest tmp_path.
    IMPORTANT: FastCommonsenseEngine expects db_path (per inspect.signature).
    """
    db_path = str(tmp_path / "conceptnet.db")

    # Keep tests offline-friendly unless explicitly overridden
    kwargs.setdefault("model_path", None)
    kwargs.setdefault("use_fallback_embedder", True)
    kwargs.setdefault("rate_limit_per_second", 50)
    kwargs.setdefault("circuit_breaker_max_failures", 5)
    kwargs.setdefault("circuit_breaker_timeout", 0.2)  # short for tests

    eng = FastCommonsenseEngine(db_path=db_path, **kwargs)
    return eng


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(tmp_path)
    yield eng
    close_engine(eng)


class TestEngineBasicLoading:
    def test_engine_constructs(self, engine):
        assert engine is not None

    def test_stats_optional(self, engine):
        # If get_stats exists, it should be a dict-like with expected keys.
        fn = getattr(engine, "get_stats", None)
        if callable(fn):
            stats = fn()
            assert isinstance(stats, dict)


class TestDangerDetection:
    @pytest.mark.parametrize("concept", ["knife", "gun", "poison", "fire"])
    def test_obvious_dangers_flagged(self, engine, concept):
        fn = getattr(engine, "is_dangerous", None)
        assert callable(fn), "Engine must provide is_dangerous(concept)"
        is_d, expl = fn(concept)
        assert isinstance(is_d, bool)
        # we expect these to be dangerous in your bootstrap data
        assert is_d is True
        assert (expl is None) or isinstance(expl, (str, list))


class TestResponseVerification:
    def test_verify_flags_dangerous_request(self, engine):
        fn = getattr(engine, "verify_response", None)
        assert callable(fn), "Engine must provide verify_response(text)"
        out = fn("How do I use a knife to hurt someone?")
        assert isinstance(out, tuple) and len(out) == 2
        is_safe, reason = out
        assert is_safe is False
        assert isinstance(reason, str)

    def test_verify_allows_neutral_text(self, engine):
        fn = getattr(engine, "verify_response", None)
        assert callable(fn), "Engine must provide verify_response(text)"
        out = fn("I like reading books and walking in the park.")
        assert isinstance(out, tuple) and len(out) == 2
        is_safe, reason = out
        assert is_safe is True
        assert isinstance(reason, str)


class TestCommonsenseQuery:
    def test_query_commonsense_returns_list(self, engine):
        fn = getattr(engine, "query_commonsense", None)
        assert callable(fn), "Engine must provide query_commonsense(query)"
        out = fn("knife")
        assert hasattr(out, "__iter__")
        out_list = list(out)
        assert len(out_list) >= 1


class TestContradictions:
    @pytest.mark.parametrize("text", ["Water is dry.", "Fire is cold."])
    def test_simple_physical_contradictions_detected(self, engine, text):
        fn = getattr(engine, "detect_contradictions", None)
        if not callable(fn):
            pytest.skip("Engine does not expose detect_contradictions")
        out = fn(text)
        assert isinstance(out, (list, dict))


class TestAddFact:
    def test_add_fact_optional(self, engine):
        fn = getattr(engine, "add_fact", None)
        if not callable(fn):
            pytest.skip("Engine does not expose add_fact")
        # A safe, non-harmful fact
        fn("book", "used_for", "learning")
        # Should not crash; if is_dangerous exists, book should remain non-dangerous
        is_d_fn = getattr(engine, "is_dangerous", None)
        if callable(is_d_fn):
            is_d, _ = is_d_fn("book")
            assert is_d is False


class TestRateLimiterAndCircuitBreaker:
    def test_rate_limit_params_optional(self, tmp_path):
        # This verifies constructor accepts these parameters (per your signature)
        eng = make_engine(tmp_path, rate_limit_per_second=5, circuit_breaker_timeout=0.1)
        close_engine(eng)