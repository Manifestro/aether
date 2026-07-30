"""Dev-tier API key store for the public preview (plan.md §5 B2).

Deliberately not a production auth system: in-memory, single-process, no
persistence. It exists so the HTTP layer has somewhere to enforce "no
anonymous access" and "cap concurrent turns per key" from day one, in a
shape a real key-management service can replace without touching
`aether_api.http`.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


class QuotaExceeded(Exception):
    """Raised when a key is already at its concurrent-turn limit."""


@dataclass
class ApiKey:
    key: str
    owner: str
    max_concurrent_turns: int = 2
    active_turns: int = field(default=0, repr=False)


class ApiKeyStore:
    def __init__(self, keys: Dict[str, ApiKey]) -> None:
        self._keys = keys

    def resolve(self, key: str) -> Optional[ApiKey]:
        return self._keys.get(key)

    @contextmanager
    def claim_turn_slot(self, api_key: ApiKey) -> Iterator[None]:
        if api_key.active_turns >= api_key.max_concurrent_turns:
            raise QuotaExceeded(f"key {api_key.owner!r} exceeded max_concurrent_turns")
        api_key.active_turns += 1
        try:
            yield
        finally:
            api_key.active_turns -= 1

    @classmethod
    def from_env(cls, var: str = "AETHER_API_KEYS") -> "ApiKeyStore":
        """Dev convenience: comma-separated `key[:owner[:limit]]` entries.

        A real deployment replaces this constructor entirely (persistent
        store, rotation, per-key rate limit) without changing callers that
        only depend on `ApiKeyStore.resolve`/`claim_turn_slot`.
        """
        raw = os.environ.get(var, "dev-key")
        keys: Dict[str, ApiKey] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            key = parts[0]
            owner = parts[1] if len(parts) > 1 else "dev"
            limit = int(parts[2]) if len(parts) > 2 else 2
            keys[key] = ApiKey(key=key, owner=owner, max_concurrent_turns=limit)
        return cls(keys)
