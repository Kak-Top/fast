"""
Kafka Configuration — Cloud Kafka (Aiven) Support
===================================================
Shared config for all Kafka producers/consumers.
Supports both local Kafka (no auth) and Aiven cloud Kafka (SSL certs).

─── For Render Deployment (RECOMMENDED) ───
Set these 3 environment variables in Render dashboard:
  KAFKA_CA_B64   = base64 of ca.pem     (run: base64 certs/ca.pem)
  KAFKA_CERT_B64 = base64 of service.cert
  KAFKA_KEY_B64  = base64 of service.key
  KAFKA_BOOTSTRAP = your-kafka.aivencloud.com:12345

─── For Local Development ───
1. Put ca.pem, service.cert, service.key inside a certs/ folder next to this file.
2. Set KAFKA_BOOTSTRAP in your .env or shell.
That's it — the local certs/ folder is auto-detected.
"""

import base64
import logging
import os
import ssl
import tempfile

log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# ─── Local cert paths (for development) ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CA   = os.path.join(BASE_DIR, "certs", "ca.pem")
DEFAULT_CERT = os.path.join(BASE_DIR, "certs", "service.cert")
DEFAULT_KEY  = os.path.join(BASE_DIR, "certs", "service.key")


def is_cloud_kafka() -> bool:
    """Returns True if SSL certs are available (Aiven cloud Kafka)."""
    # Render env vars take priority
    if os.getenv("KAFKA_CA_B64") and os.getenv("KAFKA_CERT_B64") and os.getenv("KAFKA_KEY_B64"):
        return True
    # Local certs/ folder fallback
    return (
        os.path.exists(DEFAULT_CA)
        and os.path.exists(DEFAULT_CERT)
        and os.path.exists(DEFAULT_KEY)
    )


def _get_ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context for Aiven Kafka.

    Priority:
      1. KAFKA_CA_B64 / KAFKA_CERT_B64 / KAFKA_KEY_B64  — Render env vars (base64-encoded)
      2. certs/ca.pem + certs/service.cert + certs/service.key — local development folder
    """
    ca_b64   = os.getenv("KAFKA_CA_B64")
    cert_b64 = os.getenv("KAFKA_CERT_B64")
    key_b64  = os.getenv("KAFKA_KEY_B64")

    if ca_b64 and cert_b64 and key_b64:
        # ── Render / production: decode base64 env vars into temp files ──────
        log.info("Using Kafka SSL certs from environment variables (Render)")

        tmp_dir   = tempfile.mkdtemp()
        ca_path   = os.path.join(tmp_dir, "ca.pem")
        cert_path = os.path.join(tmp_dir, "service.cert")
        key_path  = os.path.join(tmp_dir, "service.key")

        with open(ca_path,   "wb") as f:
            f.write(base64.b64decode(ca_b64))
        with open(cert_path, "wb") as f:
            f.write(base64.b64decode(cert_b64))
        with open(key_path,  "wb") as f:
            f.write(base64.b64decode(key_b64))

    else:
        # ── Local development: use certs/ directory ───────────────────────────
        log.info("Using Kafka SSL certs from local 'certs/' directory")
        ca_path   = DEFAULT_CA
        cert_path = DEFAULT_CERT
        key_path  = DEFAULT_KEY

        # Give a clear error if files are missing locally too
        for label, path in [("CA", ca_path), ("cert", cert_path), ("key", key_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Kafka SSL {label} not found at '{path}'. "
                    "Set KAFKA_CA_B64 / KAFKA_CERT_B64 / KAFKA_KEY_B64 env vars, "
                    "or put the cert files in the certs/ folder."
                )

    ctx = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=ca_path,
    )
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return ctx


def get_kafka_config() -> dict:
    """
    Returns Kafka connection kwargs for AIOKafkaConsumer / AIOKafkaProducer.
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
    else:
        log.warning(
            "No Kafka SSL certs found — connecting without SSL. "
            "This will fail against Aiven. Set KAFKA_CA_B64 / KAFKA_CERT_B64 / KAFKA_KEY_B64."
        )

    return config
