"""End-to-end encryption of payloads: the relay forwards ciphertext it cannot read or forge.

Clients and workers share one secret (``DISTLLM_PAYLOAD_KEY``). Every task payload and
result is sealed with ChaCha20-Poly1305 before it leaves the process:

    b"DLE1" | 12-byte random nonce | ciphertext + 16-byte tag

The relay still sees what it needs to route: roles, the session id, sizes, timing and the
``stop_chain`` flag (sent as a URL parameter). It can drop, delay or replay a sealed
message, but not read or change one: a modified message fails authentication.
"""

import base64
import binascii
import os
import secrets
from typing import Optional

MAGIC = b"DLE1"
NONCE_SIZE = 12
KEY_ENV = "DISTLLM_PAYLOAD_KEY"
# Passphrases are stretched with scrypt. The salt is fixed because every node must derive
# the same key from the same passphrase; a generated key (`distllm keygen`) skips this.
_SCRYPT_SALT = b"distllm-payload-key-v1"


class PayloadDecryptionError(ValueError):
    """A payload could not be opened: wrong key, tampered data, or encryption mismatch."""


def generate_key() -> str:
    """A fresh random key in the form accepted by ``DISTLLM_PAYLOAD_KEY``."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")


def _derive_key(secret: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
        if len(raw) == 32:
            return raw
    except (binascii.Error, ValueError):
        pass
    if len(secret) < 12:
        raise ValueError(f"{KEY_ENV} is too short: use `distllm keygen` or a passphrase of 12+ characters")
    import hashlib

    return hashlib.scrypt(secret.encode(), salt=_SCRYPT_SALT, n=2**15, r=8, p=1, maxmem=64 * 2**20, dklen=32)


class PayloadCipher:
    """Seals and opens payload bytes with a shared key."""

    def __init__(self, secret: str):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("Payload encryption needs `pip install distllm[encryption]`") from e
        self._aead = ChaCha20Poly1305(_derive_key(secret))

    def seal(self, data: bytes) -> bytes:
        nonce = os.urandom(NONCE_SIZE)
        return MAGIC + nonce + self._aead.encrypt(nonce, data, MAGIC)

    def open(self, data: bytes) -> bytes:
        if not is_sealed(data):
            # Refuse plaintext: otherwise whoever controls the relay could inject payloads.
            raise PayloadDecryptionError("Received an unencrypted payload while encryption is on")
        nonce = data[len(MAGIC):len(MAGIC) + NONCE_SIZE]
        try:
            return self._aead.decrypt(nonce, data[len(MAGIC) + NONCE_SIZE:], MAGIC)
        except Exception as e:  # cryptography raises InvalidTag
            raise PayloadDecryptionError("Payload failed authentication (wrong key or tampered data)") from e


def is_sealed(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def cipher_from(secret: Optional[str]) -> Optional[PayloadCipher]:
    """The cipher for an explicit secret, else for ``$DISTLLM_PAYLOAD_KEY``, else None."""
    secret = secret or os.environ.get(KEY_ENV)
    return PayloadCipher(secret) if secret else None
