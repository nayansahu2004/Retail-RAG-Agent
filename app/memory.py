"""In-process short-term conversation memory, keyed by session identifier."""

import threading
import time
from collections import deque


class SessionStore:
    """Thread-safe, bounded, TTL-expiring per-session chat history.

    One turn is counted as a user message plus the corresponding assistant
    reply. History older than ``ttl_s`` is purged on access.
    """

    def __init__(self, max_turns: int = 4, ttl_s: int = 3600):
        self.max_messages = max_turns * 2
        self.ttl_s = ttl_s
        self._sessions: dict[str, tuple[float, deque]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        """Return the stored history for ``session_id`` (empty if unknown or expired)."""
        with self._lock:
            self._purge_expired()
            entry = self._sessions.get(session_id)
            return list(entry[1]) if entry else []

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Append a user/assistant message pair to the session history."""
        with self._lock:
            _, history = self._sessions.get(session_id, (0.0, deque(maxlen=self.max_messages)))
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            self._sessions[session_id] = (time.monotonic(), history)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, (last_seen, _) in self._sessions.items() if now - last_seen > self.ttl_s]
        for sid in expired:
            del self._sessions[sid]
