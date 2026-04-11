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
5. In Render: store the FILE CONTENTS as env vars (see below)
6. Topics: create vitals.raw, labs.results, inference.output

Env vars to set in Render:
  KAFKA_BOOTSTRAP=your-kafka.aivencloud.com:12345
  KAFKA_SSL_CA=<contents of ca.pem>
  KAFKA_SSL_CERT=<contents of service.cert>
  KAFKA_SSL_KEY=<contents of service.key>

OR for local Kafka (no auth):
  KAFKA_BOOTSTRAP=localhost:9092
  (leave the SSL vars empty)
"""

import os
import ssl
import tempfile

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# Aiven SSL certificate contents (copy-paste from downloaded files)
KAFKA_SSL_CA = os.getenv("KAFKA_SSL_CA", "")
KAFKA_SSL_CERT = os.getenv("KAFKA_SSL_CERT", "")
KAFKA_SSL_KEY = os.getenv("KAFKA_SSL_KEY", "")

# Aiven SSL certificate file paths (alternative: point to files directly)
KAFKA_SSL_CA_PATH = os.getenv("KAFKA_SSL_CA_PATH", "")
KAFKA_SSL_CERT_PATH = os.getenv("KAFKA_SSL_CERT_PATH", "")
KAFKA_SSL_KEY_PATH = os.getenv("KAFKA_SSL_KEY_PATH", "")

# Cache for temp file paths (created from env var contents)
_temp_files = {}


def _write_temp_file(name: str, content: str) -> str:
    """Write content to a temp file and return its path. Cached per name."""
    if name in _temp_files:
        return _temp_files[name]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{name}.pem", delete=False, prefix="kafka_"
    )
    tmp.write(content)
    tmp.flush()
    tmp.close()
    _temp_files[name] = tmp.name
    return tmp.name


def is_cloud_kafka() -> bool:
    """Returns True if we have SSL certs configured (Aiven cloud Kafka)."""
    has_content = bool(KAFKA_SSL_CA and KAFKA_SSL_CERT and KAFKA_SSL_KEY)
    has_paths = bool(KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH)
    return has_content or has_paths


def _get_ssl_context() -> ssl.SSLContext:
    """Build SSL context from either env var contents or file paths."""
    # Determine file paths
    if KAFKA_SSL_CA_PATH and KAFKA_SSL_CERT_PATH and KAFKA_SSL_KEY_PATH:
        ca_path = KAFKA_SSL_CA_PATH
        cert_path = KAFKA_SSL_CERT_PATH
        key_path = KAFKA_SSL_KEY_PATH
    else:
        # Write env var contents to temp files (aiokafka needs file paths)
        ca_path = _write_temp_file("ca", KAFKA_SSL_CA)
        cert_path = _write_temp_file("cert", KAFKA_SSL_CERT)
        key_path = _write_temp_file("key", KAFKA_SSL_KEY)

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
