import time

from app.memory import SessionStore


def test_keeps_only_last_turns():
    store = SessionStore(max_turns=1, ttl_s=60)
    store.append("s", "q1", "a1")
    store.append("s", "q2", "a2")
    assert [m["content"] for m in store.get("s")] == ["q2", "a2"]


def test_sessions_expire():
    store = SessionStore(max_turns=2, ttl_s=0)
    store.append("s", "q", "a")
    time.sleep(0.01)
    assert store.get("s") == []
