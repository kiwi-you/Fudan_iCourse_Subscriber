/**
 * OpenSSL-compatible AES-256-CBC encryption/decryption using Web Crypto API.
 *
 * File format (same as `openssl enc -aes-256-cbc -salt -pbkdf2`):
 *   Bytes  0-7:  "Salted__"  (magic header)
 *   Bytes  8-15: 8-byte random salt
 *   Bytes 16+:   AES-256-CBC ciphertext (PKCS7 padded)
 *
 * Two key flavors:
 *   v2:     password = sha256("ICSv2:" + stuid + ":" + uispsw)  (hex)
 *           PBKDF2 iterations: 100000
 *   legacy: password = stuid + uispsw + dashscope + smtp        (concat)
 *           PBKDF2 iterations: 10000
 *
 * Derivation:
 *   PBKDF2-HMAC-SHA256(password, salt, iterations, dkLen=48)
 *   -> first 32 bytes = AES key, last 16 bytes = IV
 */

window.ICS = window.ICS || {};

var MAGIC = new TextEncoder().encode("Salted__");
var NEW_ITERATIONS = 100000;
var LEGACY_ITERATIONS = 10000;

function _checkWebCrypto() {
  if (!window.crypto || !window.crypto.subtle) {
    throw new Error(
      "Web Crypto API is not available. Please access this page via HTTPS (GitHub Pages) or http://localhost. " +
      "Current protocol: " + location.protocol + " host: " + location.host
    );
  }
}

async function _sha256Hex(text) {
  _checkWebCrypto();
  var bytes = new TextEncoder().encode(text);
  var digest = await window.crypto.subtle.digest("SHA-256", bytes);
  var arr = new Uint8Array(digest);
  return Array.from(arr).map(function (b) {
    return b.toString(16).padStart(2, "0");
  }).join("");
}

async function _deriveKeyAndIV(password, salt, iterations) {
  _checkWebCrypto();
  var enc = new TextEncoder();
  var baseKey = await window.crypto.subtle.importKey(
    "raw", enc.encode(password), "PBKDF2", false, ["deriveBits"]
  );
  var bits = await window.crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: salt, iterations: iterations, hash: "SHA-256" },
    baseKey, 48 * 8
  );
  var key = await window.crypto.subtle.importKey(
    "raw", bits.slice(0, 32), { name: "AES-CBC" }, false, ["encrypt", "decrypt"]
  );
  return { key: key, iv: new Uint8Array(bits.slice(32, 48)) };
}

async function _icsDecrypt(encryptedBytes, password, iterations) {
  iterations = iterations || NEW_ITERATIONS;
  var headerStr = new TextDecoder().decode(encryptedBytes.slice(0, 8));
  if (headerStr !== "Salted__") {
    throw new Error("Invalid file: missing OpenSSL 'Salted__' header");
  }
  var salt = encryptedBytes.slice(8, 16);
  var ciphertext = encryptedBytes.slice(16);
  var derived = await _deriveKeyAndIV(password, salt, iterations);
  var plainBuffer = await window.crypto.subtle.decrypt(
    { name: "AES-CBC", iv: derived.iv }, derived.key, ciphertext
  );
  return new Uint8Array(plainBuffer);
}

async function _icsEncrypt(plainBytes, password, iterations) {
  iterations = iterations || NEW_ITERATIONS;
  _checkWebCrypto();
  var salt = window.crypto.getRandomValues(new Uint8Array(8));
  var derived = await _deriveKeyAndIV(password, salt, iterations);
  var cipherBuffer = await window.crypto.subtle.encrypt(
    { name: "AES-CBC", iv: derived.iv }, derived.key, plainBytes
  );
  var cipherBytes = new Uint8Array(cipherBuffer);
  var result = new Uint8Array(MAGIC.length + salt.length + cipherBytes.length);
  result.set(MAGIC, 0);
  result.set(salt, MAGIC.length);
  result.set(cipherBytes, MAGIC.length + salt.length);
  return result;
}

async function _icsBuildPasswordV2(secrets) {
  return await _sha256Hex("ICSv2:" + secrets.stuid + ":" + secrets.uispsw);
}

function _icsBuildPasswordLegacy(secrets) {
  return secrets.stuid + secrets.uispsw +
         (secrets.dashscope || "") + (secrets.smtp || "");
}

async function _icsDecryptWithFallback(encryptedBytes, secrets) {
  try {
    var pwV2 = await _icsBuildPasswordV2(secrets);
    return {
      data: await _icsDecrypt(encryptedBytes, pwV2, NEW_ITERATIONS),
      version: "v2",
    };
  } catch (e) {
    var pwLegacy = _icsBuildPasswordLegacy(secrets);
    return {
      data: await _icsDecrypt(encryptedBytes, pwLegacy, LEGACY_ITERATIONS),
      version: "legacy",
    };
  }
}

window.ICS.crypto = {
  decrypt: _icsDecrypt,
  encrypt: _icsEncrypt,
  buildPassword: _icsBuildPasswordV2,
  buildPasswordV2: _icsBuildPasswordV2,
  buildPasswordLegacy: _icsBuildPasswordLegacy,
  decryptWithFallback: _icsDecryptWithFallback,
  NEW_ITERATIONS: NEW_ITERATIONS,
  LEGACY_ITERATIONS: LEGACY_ITERATIONS,
};
