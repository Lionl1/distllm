"""Bearer token specs: one token, several (rotation), or a file the relay re-reads.

A spec is what ``--token`` / ``DISTLLM_TOKEN`` hold:

- ``s3cret`` — one token;
- ``new,old`` — any of them is accepted (rotate without downtime: add the new token,
  move clients and workers to it, then drop the old one);
- ``@/path/tokens`` — tokens from a file (one per line or comma-separated, ``#`` comments).
  The relay re-reads the file when it changes, so tokens rotate without a restart.

Clients and workers send the first token of their spec.
"""

import os
import secrets
import time
from typing import List, Optional

RELOAD_INTERVAL = 1.0  # seconds between checks of a token file


def parse_tokens(text: str) -> List[str]:
    tokens = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        tokens.extend(t.strip() for t in line.split(",") if t.strip())
    return tokens


def _read_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return parse_tokens(f.read())


def first_token(spec: Optional[str]) -> Optional[str]:
    """The token a client or worker sends for ``spec``."""
    if not spec:
        return None
    tokens = _read_file(spec[1:]) if spec.startswith("@") else parse_tokens(spec)
    return tokens[0] if tokens else None


class TokenSet:
    """Tokens accepted by one side of the relay; ``None`` spec = no authentication."""

    def __init__(self, spec: Optional[str]):
        self.spec = spec or None
        self.path = self.spec[1:] if self.spec and self.spec.startswith("@") else None
        self._tokens: List[bytes] = [] if self.path else [t.encode() for t in parse_tokens(self.spec or "")]
        self._mtime: Optional[float] = None
        self._checked = float("-inf")
        if self.path:
            self._reload()

    @property
    def enabled(self) -> bool:
        return self.spec is not None

    def _reload(self) -> None:
        self._checked = time.monotonic()
        try:
            mtime = os.stat(self.path).st_mtime_ns
            if mtime != self._mtime:
                self._tokens = [t.encode() for t in _read_file(self.path)]
                self._mtime = mtime
        except OSError as e:
            # Keep the last good tokens; with none loaded yet everything is refused.
            print(f"[!] Cannot read token file {self.path}: {e}")

    def accepts(self, value: str) -> bool:
        if self.path and time.monotonic() - self._checked >= RELOAD_INTERVAL:
            self._reload()
        given = value.encode()
        # Compare with every token (no early exit) in constant time per comparison.
        return sum(secrets.compare_digest(given, t) for t in self._tokens) > 0
