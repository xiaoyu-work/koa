"""Tests for credential encryption."""

import pytest

from koa.credentials.encryption import CredentialEncryptor


class TestCredentialEncryptor:
    def test_encrypt_decrypt_roundtrip(self):
        enc = CredentialEncryptor("test-secret-key-that-is-32bytes!")
        original = {"access_token": "abc123", "refresh_token": "xyz789"}
        encrypted = enc.encrypt(original)
        assert encrypted != original
        assert isinstance(encrypted, str)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_produces_different_output_each_time(self):
        enc = CredentialEncryptor("test-secret-key-that-is-32bytes!")
        data = {"token": "abc"}
        e1 = enc.encrypt(data)
        e2 = enc.encrypt(data)
        assert e1 != e2

    def test_decrypt_with_wrong_key_fails(self):
        enc1 = CredentialEncryptor("key-one-that-is-exactly-32bytes!")
        enc2 = CredentialEncryptor("key-two-that-is-exactly-32bytes!")
        encrypted = enc1.encrypt({"secret": "data"})
        with pytest.raises(Exception):
            enc2.decrypt(encrypted)

    def test_noop_when_no_key(self):
        enc = CredentialEncryptor(None)
        data = {"token": "abc123"}
        assert enc.encrypt(data) == data
        assert enc.decrypt(data) == data

    def test_decrypt_plaintext_dict_passthrough(self):
        enc = CredentialEncryptor("test-secret-key-that-is-32bytes!")
        data = {"token": "abc123"}
        result = enc.decrypt(data)
        assert result == data
