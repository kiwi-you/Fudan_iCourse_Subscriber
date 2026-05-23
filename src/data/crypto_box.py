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
from typing import Callable

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


# ── Plaintext sanity validators ────────────────────────────────────────
# AES-CBC + PKCS7 has a ~1/256 chance of accepting a wrong key (the last
# byte of the garbled plaintext happens to be 0x01).  Without a sanity
# check the random bytes get propagated downstream and either crash with
# an unhelpful error or, worse, malfunction silently.  Pass one of these
# to ``decrypt_with_fallback(validate=...)`` to reject false positives so
# the next key is tried instead.

def is_sqlite(pt: bytes) -> bool:
    """Legacy single-file uncompressed sqlite db."""
    return len(pt) >= 16 and pt[:16] == b"SQLite format 3" + bytes([0])


def is_gzip(pt: bytes) -> bool:
    """Gzipped sqlite (legacy compressed db, or a decrypted shard)."""
    return len(pt) >= 2 and pt[:2] == bytes([0x1F, 0x8B])


def is_json_obj(pt: bytes) -> bool:
    """Encrypted index plaintext is a JSON object/array."""
    s = pt.lstrip()
    return bool(s) and s[:1] in (b"{", b"[")


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
            iterations: int = NEW_ITERATIONS,
            deterministic: bool = False) -> bytes:
    """Encrypt with random salt by default. Returns full OpenSSL envelope.

    With ``deterministic=True`` the salt is derived from
    ``sha256(plaintext)[:8]`` instead of ``os.urandom(8)``.  Same
    ``(password, plaintext)`` then always produces the same ciphertext, so
    unchanged content keeps its git blob sha across deploys — letting the
    frontend's content-addressed shard cache actually hit.  Used by sharder
    for shard files + index.
    """
    if deterministic:
        salt = hashlib.sha256(plaintext).digest()[:SALT_SIZE]
    else:
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
                          smtp: str = "",
                          validate: Callable[[bytes], bool] | None = None,
                          ) -> tuple[bytes, str]:
    """Try v2 password first, fall back to legacy.

    Returns (plaintext, version_used) where version_used is "v2" or "legacy".
    Caller can use the version to decide whether to re-encrypt with the new key.

    If ``validate`` is given, the decrypted plaintext must satisfy
    ``validate(pt) -> True``; otherwise the v2 attempt is rejected and the
    legacy key is tried.  If the legacy key also fails validation, raises
    ValueError.  Without a validator a wrong key has ~1/256 chance of
    decrypting to garbage with valid PKCS7 padding — the validator catches
    that case using a magic-byte check.

    Raises ValueError if neither password produces a valid plaintext.
    """
    new_pw = derive_new_password(stuid, uispsw)
    try:
        pt = decrypt(blob, new_pw, NEW_ITERATIONS)
        if validate is None or validate(pt):
            return pt, "v2"
    except Exception:
        pass
    legacy_pw = derive_legacy_password(stuid, uispsw, dashscope, smtp)
    pt = decrypt(blob, legacy_pw, LEGACY_ITERATIONS)
    if validate is not None and not validate(pt):
        raise ValueError(
            "decryption produced bytes that did not pass plaintext validation"
        )
    return pt, "legacy"
