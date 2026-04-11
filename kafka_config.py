"""
Kafka Configuration — Cloud Kafka (Aiven) Support
===================================================
Shared config for all Kafka producers/consumers.
Supports both local Kafka (no auth) and Aiven cloud Kafka (SSL certs).

─── For Aiven Free Tier ───
1. Sign up at https://console.aiven.io (free, no credit card)
2. Create project → Create Aiven for Apache Kafka → Free plan
3. Go to service Overview → Connection information
4. Download 3 files: ca.pem, service.cert, service.key

─── How to set env vars in Render ───
The PEM file contents have newlines that get mangled in env vars.
You MUST base64-encode them first:

  On Linux/Mac:
    base64 -w 0 ca.pem          → copy output → paste as KAFKA_SSL_CA_B64
    base64 -w 0 service.cert    → copy output → paste as KAFKA_SSL_CERT_B64
    base64 -w 0 service.key     → copy output → paste as KAFKA_SSL_KEY_B64

  On Windows (PowerShell):
    [Convert]::ToBase64String([IO.File]::ReadAllBytes("ca.pem"))
    [Convert]::ToBase64String([IO.File]::ReadAllBytes("service.cert"))
    [Convert]::ToBase64String([IO.File]::ReadAllBytes("service.key"))

Env vars to set in Render:
  KAFKA_BOOTSTRAP=your-kafka.aivencloud.com:12345
  KAFKA_SSL_CA_B64=<base64 of ca.pem>
  KAFKA_SSL_CERT_B64=<base64 of service.cert>
  KAFKA_SSL_KEY_B64=<base64 of service.key>
"""

import base64
import logging
import os
import ssl
import tempfile

log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# Base64-encoded cert contents (recommended for Render)
KAFKA_SSL_CA_B64 = os.getenv("KAFKA_SSL_CA_B64", "")
KAFKA_SSL_CERT_B64 = os.getenv("KAFKA_SSL_CERT_B64", "")
KAFKA_SSL_KEY_B64 = os.getenv("KAFKA_SSL_KEY_B64", "")

# Direct file paths (alternative: if you commit cert files to repo)
KAFKA_SSL_CA_PATH = os.getenv("KAFKA_SSL_CA_PATH", "")
KAFKA_SSL_CERT_PATH = os.getenv("KAFKA_SSL_CERT_PATH", "")
KAFKA_SSL_KEY_PATH = os.getenv("KAFKA_SSL_KEY_PATH", "")

# Cache for temp file paths
_temp_files = {}


def _decode_and_write(name: str, b64_content: str) -> str:
    """Decode base64 content, write to temp file, return path. Cached."""
    if name in _temp_files:
        return _temp_files[name]

    raw = base64.b64decode(b64_content)
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=f"_{name}.pem", delete=False, prefix="kafka_"
    )
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    _temp_files[name] = tmp.name
    log.info("Wrote Kafka SSL %s to %s (%d bytes)", name, tmp.name, len(raw))
    return tmp.name


def is_cloud_kafka() -> bool:
    """Returns True if we have SSL certs configured (Aiven cloud Kafka)."""
    has_b64 = bool(KAFKA_SSL_CA_B64 and KAFKA_SSL_CERT_B64 and KAFKA_SSL_KEY_B64)
    has_paths = bool(KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH)
    return has_b64 or has_paths


def _get_ssl_context() -> ssl.SSLContext:
    """Build SSL context from either base64 env vars or file paths."""
    # Determine file paths
    if KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH:
        ca_path = KAFKA_SSL_CA_PATH
        cert_path = KAFKA_SSL_CERT_PATH
        key_path = KAFKA_SSL_KEY_PATH
        log.info("Using Kafka SSL certs from file paths")
    else:
        # Decode base64 env vars to temp files
        ca_path = _decode_and_write("ca", KAFKA_SSL_CA_B64)
        cert_path = _decode_and_write("cert", KAFKA_SSL_CERT_B64)
        key_path = _decode_and_write("key", KAFKA_SSL_KEY_B64)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(ca_path)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
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
