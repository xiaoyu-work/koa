"""Credential encryption for at-rest protection.

Uses AES-256-GCM via the cryptography library.
When KOA_CREDENTIAL_KEY is set, credentials are encrypted before storing
in PostgreSQL and decrypted on retrieval.

When no key is configured, operates as a passthrough (no encryption).
This allows gradual migration: existing plaintext credentials remain
readable, and new credentials are encrypted once a key is set.
"""

import base64
import hashlib
import json
import logging
import secrets
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

_ENCRYPTED_PREFIX = "enc:v1:"


class CredentialEncryptor:
    """Encrypt/decrypt credential dicts using AES-256-GCM.

    Args:
        key: Encryption key string. Hashed to 32 bytes with SHA-256.
             If None, operates as a no-op passthrough.
    """

    def __init__(self, key: Optional[str] = None):
        if key and _HAS_CRYPTO:
            self._key = hashlib.sha256(key.encode()).digest()
            self._aesgcm = AESGCM(self._key)
            self._enabled = True
        else:
            self._key = None
            self._aesgcm = None
            self._enabled = False
            if key and not _HAS_CRYPTO:
                logger.warning(
                    "KOA_CREDENTIAL_KEY is set but 'cryptography' package "
                    "is not installed. Credentials will NOT be encrypted. "
                    "Install with: pip install cryptography"
                )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def encrypt(self, data: Union[dict, str]) -> Union[str, dict]:
        """Encrypt a credential dict to an encrypted string."""
        if not self._enabled:
            return data

        plaintext = json.dumps(data).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        payload = base64.b64encode(nonce + ciphertext).decode("ascii")
        return f"{_ENCRYPTED_PREFIX}{payload}"

    def decrypt(self, data: Union[str, dict]) -> dict:
        """Decrypt an encrypted string back to a credential dict.

        If data is already a plain dict (unencrypted legacy), returns it as-is.
        """
        if isinstance(data, dict):
            return data

        if not isinstance(data, str):
            return data

        if not data.startswith(_ENCRYPTED_PREFIX):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data

        if not self._enabled:
            raise ValueError(
                "Encrypted credentials found but KOA_CREDENTIAL_KEY is not set."
            )

        payload = data[len(_ENCRYPTED_PREFIX):]
        raw = base64.b64decode(payload)
        nonce = raw[:12]
        ciphertext = raw[12:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))