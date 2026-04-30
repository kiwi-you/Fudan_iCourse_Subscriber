"""AES-256-CBC + PBKDF2-SHA256 envelope, OpenSSL `enc -salt -pbkdf2` compatible.

File layout:
  bytes  0..7   : "Salted__"
  bytes  8..15  : 8-byte random salt
  bytes 16..    : AES-256-CBC PKCS7 ciphertext

Key derivation:
  PBKDF2-HMAC-SHA256(password, salt, iterations, dkLen=48)
  -> first 32 bytes = AES-256 key
  -> last 16 bytes = IV

Two key flavors:
  - new (v2):  sha256("ICSv2:" + stuid + ":" + uispsw).hexdigest(),  100_000 iter
  - legacy:    stuid + uispsw + dashscope + smtp                ,    10_000 iter
"""

from __future__ import annotations

import hashlib
import os

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad

MAGIC = b"Salted__"
SALT_SIZE = 8
KEY_SIZE = 32
IV_SIZE = 16
KEY_IV_SIZE = KEY_SIZE + IV_SIZE  # 48
NEW_ITERATIONS = 100_000
LEGACY_ITERATIONS = 10_000


def derive_new_password(stuid: str, uispsw: str) -> str:
    """Derive the v2 password string from stuid + uispsw.

    Stable, deterministic, 64 hex chars. Used as the input to PBKDF2.
    """
    raw = f"ICSv2:{stuid}:{uispsw}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derive_legacy_password(stuid: str, uispsw: str,
                           dashscope: str, smtp: str) -> str:
    """Derive the legacy password string (4 concatenated env values)."""
    return f"{stuid}{uispsw}{dashscope}{smtp}"


def _derive_key_iv(password: str, salt: bytes,
                   iterations: int) -> tuple[bytes, bytes]:
    keyiv = PBKDF2(
        password.encode("utf-8"),
        salt,
        dkLen=KEY_IV_SIZE,
        count=iterations,
        hmac_hash_module=SHA256,
    )
    return keyiv[:KEY_SIZE], keyiv[KEY_SIZE:]


def encrypt(plaintext: bytes, password: str,
            iterations: int = NEW_ITERATIONS) -> bytes:
    """Encrypt with random salt. Returns full OpenSSL envelope."""
    salt = os.urandom(SALT_SIZE)
    key, iv = _derive_key_iv(password, salt, iterations)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(plaintext, AES.block_size))
    return MAGIC + salt + ct


def decrypt(blob: bytes, password: str,
            iterations: int = NEW_ITERATIONS) -> bytes:
    """Decrypt an OpenSSL envelope. Raises ValueError on bad header / padding."""
    if len(blob) < len(MAGIC) + SALT_SIZE + AES.block_size:
        raise ValueError("blob too short to be a valid OpenSSL envelope")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("missing 'Salted__' header")
    salt = blob[len(MAGIC) : len(MAGIC) + SALT_SIZE]
    ct = blob[len(MAGIC) + SALT_SIZE :]
    key, iv = _derive_key_iv(password, salt, iterations)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)


def decrypt_with_fallback(blob: bytes, *, stuid: str, uispsw: str,
                          dashscope: str = "",
                          smtp: str = "") -> tuple[bytes, str]:
    """Try v2 password first, fall back to legacy.

    Returns (plaintext, version_used) where version_used is "v2" or "legacy".
    Caller can use the version to decide whether to re-encrypt with the new key.

    Raises ValueError if neither password succeeds.
    """
    new_pw = derive_new_password(stuid, uispsw)
    try:
        return decrypt(blob, new_pw, NEW_ITERATIONS), "v2"
    except (ValueError, Exception):
        pass
    legacy_pw = derive_legacy_password(stuid, uispsw, dashscope, smtp)
    return decrypt(blob, legacy_pw, LEGACY_ITERATIONS), "legacy"
