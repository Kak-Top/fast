"""
Kafka Configuration — Cloud Kafka (Aiven) Support
===================================================
Shared config for all Kafka producers/consumers.
Supports both local Kafka (no auth) and Aiven cloud Kafka (SSL certs).

─── For Aiven Free Tier (THE EASIEST WAY) ───
1. Download `ca.pem`, `service.cert`, and `service.key` from Aiven.
2. In your repo, create a folder named `certs` next to this file.
3. Put the 3 files inside the `certs` folder.
4. Git commit and push them to Render.
That's it. It will automatically detect them.

Env vars to set in Render:
  KAFKA_BOOTSTRAP=your-kafka.aivencloud.com:12345
  (No need to set SSL env vars if you put the files in the certs folder!)
"""

import base64
import logging
import os
import ssl
import tempfile

log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# ─── 1. Default local cert paths (Easiest Method) ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CA = os.path.join(BASE_DIR, "certs", "ca.pem")
DEFAULT_CERT = os.path.join(BASE_DIR, "certs", "service.cert")
DEFAULT_KEY = os.path.join(BASE_DIR, "certs", "service.key")

# ─── 2. Base64-encoded cert contents (Fallback Method) ───
KAFKA_SSL_CA_B64 = os.getenv("KAFKA_SSL_CA_B64", "")
KAFKA_SSL_CERT_B64 = os.getenv("KAFKA_SSL_CERT_B64", "")
KAFKA_SSL_KEY_B64 = os.getenv("KAFKA_SSL_KEY_B64", "")

# ─── 3. Direct file paths (Alternative Fallback) ───
KAFKA_SSL_CA_PATH = os.getenv("KAFKA_SSL_CA_PATH", "")
KAFKA_SSL_CERT_PATH = os.getenv("KAFKA_SSL_CERT_PATH", "")
KAFKA_SSL_KEY_PATH = os.getenv("KAFKA_SSL_KEY_PATH", "")

_temp_files = {}


def _decode_and_write(name: str, b64_content: str) -> str:
    if name in _temp_files:
        return _temp_files[name]
    raw = base64.b64decode(b64_content)
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=f"_{name}.pem", delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    _temp_files[name] = tmp.name
    return tmp.name


def is_cloud_kafka() -> bool:
    """Returns True if we have SSL certs configured (Aiven cloud Kafka)."""
    if os.path.exists(DEFAULT_CA) and os.path.exists(DEFAULT_CERT) and os.path.exists(DEFAULT_KEY):
        return True
    has_b64 = bool(KAFKA_SSL_CA_B64 and KAFKA_SSL_CERT_B64 and KAFKA_SSL_KEY_B64)
    has_paths = bool(KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH)
    return has_b64 or has_paths


def _get_ssl_context() -> ssl.SSLContext:
    # 1. Try local certs folder first
    if os.path.exists(DEFAULT_CA) and os.path.exists(DEFAULT_CERT) and os.path.exists(DEFAULT_KEY):
        ca_path, cert_path, key_path = DEFAULT_CA, DEFAULT_CERT, DEFAULT_KEY
        log.info("Using Kafka SSL certs from local 'certs/' directory")
    # 2. Try explicit paths
    elif KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH:
        ca_path, cert_path, key_path = KAFKA_SSL_CA_PATH, KAFKA_SSL_CERT_PATH, KAFKA_SSL_KEY_PATH
        log.info("Using Kafka SSL certs from env var file paths")
    # 3. Try Base64 strings
    elif KAFKA_SSL_CA_B64:
        ca_path = _decode_and_write("ca", KAFKA_SSL_CA_B64)
        cert_path = _decode_and_write("cert", KAFKA_SSL_CERT_B64)
        key_path = _decode_and_write("key", KAFKA_SSL_KEY_B64)
        log.info("Using Kafka SSL certs decoded from B64 env vars")
    else:
        raise RuntimeError("Cloud Kafka enabled but no SSL certs found")

    # Use default context which sets up SECURE defaults (including SNI)
    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=ca_path)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return ctx


def get_kafka_config() -> dict:
    """
    Returns kafka connection kwargs for AIOKafkaConsumer / AIOKafkaProducer.
    Works with both local and Aiven cloud Kafka.

    Usage:
        from kafka_config import get_kafka_config
        consumer = AIOKafkaConsumer("topic", **get_kafka_config())
        producer = AIOKafkaProducer(**get_kafka_config())
    """
    config = {
        "bootstrap_servers": KAFKA_BOOTSTRAP,
    }

    if is_cloud_kafka():
        config.update({
            "security_protocol": "SSL",
            "ssl_context": _get_ssl_context(),
        })

    return config
