"""
TEE Gateway Middleware
======================
Intercepts ALL requests to your FastAPI.
Runs anomaly detection, adds threat info to responses,
blocks high-threat requests.
Logs every request to the immutable Merkle audit trail.
NO SIEM dependencies.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from typing import Callable

logger = logging.getLogger("tee.gateway")

# ── Configuration (all via environment variables) ──────────────────
TEE_ENABLED = os.getenv("TEE_ENABLED", "true").lower() == "true"
TEE_BLOCK_THRESHOLD = float(os.getenv("TEE_BLOCK_THRESHOLD", "0.8"))
TEE_SKIP_PATHS = os.getenv(
    "TEE_SKIP_PATHS",
    "/docs,/openapi.json,/redoc,/health,/tee/health"
).split(",")
TEE_MONITOR_ONLY_PATHS = os.getenv(
    "TEE_MONITOR_ONLY_PATHS",
    "/tee/,/ai/"
).split(",")


class TEEGatewayMiddleware(BaseHTTPMiddleware):
    """
    TEE Gateway Middleware — intercepts every request.

    For each request:
    1. Skip if path is in TEE_SKIP_PATHS
    2. Extract request features
    3. Run anomaly detection
    4. Log to Merkle audit trail (immutable)
    5. If threat_score >= TEE_BLOCK_THRESHOLD → BLOCK (403)
    6. If threat_score < threshold → ALLOW, add threat info to response headers
    7. Add security headers to all responses
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── Skip if TEE is disabled ──────────────────────────────
        if not TEE_ENABLED:
            return await call_next(request)

        # ── Skip docs/health paths ───────────────────────────────
        path = request.url.path
        if any(path.startswith(skip) for skip in TEE_SKIP_PATHS):
            return await call_next(request)

        start_time = time.time()

        try:
            # ── Run Anomaly Detection ─────────────────────────────
            from services.anomaly import get_detector
            detector = get_detector()

            # Build request info for detection
            request_info = {
                "endpoint": path,
                "method": request.method,
                "user": self._extract_user(request),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": self._extract_ip(request),
                "request_data": {},
            }

            # Run detection
            threat_result = detector.detect(request_info)
            threat_score = threat_result["model_output"]["threat_score"]
            threat_type = threat_result["model_output"]["threat_type"]

            elapsed = time.time() - start_time
            logger.info(
                f"TEE Gateway: {request.method} {path} | "
                f"threat={threat_score:.4f} ({threat_type}) | "
                f"user={request_info['user']} | "
                f"time={elapsed*1000:.1f}ms"
            )

            # ── Log to Merkle Audit Trail ────────────────────────
            try:
                from services.merkle_audit import get_merkle_tree
                merkle = get_merkle_tree()
                merkle.add_entry(
                    event_type="REQUEST_SCANNED",
                    actor=request_info["user"],
                    data={
                        "endpoint": path,
                        "method": request.method,
                        "threat_score": threat_score,
                        "threat_type": threat_type,
                        "action": "BLOCKED" if threat_score >= TEE_BLOCK_THRESHOLD else "ALLOWED",
                    },
                )
            except Exception as me:
                logger.error(f"Merkle audit logging failed: {me}")

            # ── Block or Allow ────────────────────────────────────
            is_monitor_only = any(
                path.startswith(m) for m in TEE_MONITOR_ONLY_PATHS
            )

            if threat_score >= TEE_BLOCK_THRESHOLD and not is_monitor_only:
                logger.warning(
                    f"🚫 BLOCKED: {request.method} {path} | "
                    f"threat={threat_score:.4f} ({threat_type}) | "
                    f"user={request_info['user']}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Request blocked by TEE Gateway — high threat detected",
                        "threat_score": threat_score,
                        "threat_type": threat_type,
                        "proof": threat_result["proof"],
                        "blocked_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # ── Allow: Add threat info to request state ───────────
            request.state.threat_assessment = threat_result
            request.state.threat_score = threat_score

            # Call next middleware / route handler
            response = await call_next(request)

            # ── Add Security Headers ──────────────────────────────
            response.headers["X-TEE-Threat-Score"] = str(threat_score)
            response.headers["X-TEE-Threat-Type"] = threat_type
            response.headers["X-TEE-Proof"] = threat_result["proof"]
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-TEE-Version"] = "1.0.0"
            response.headers["X-Response-Time-Ms"] = f"{(time.time() - start_time)*1000:.1f}"

            return response

        except Exception as e:
            logger.error(f"TEE Gateway error: {e}")
            # On error: allow request through (fail-open for availability)
            response = await call_next(request)
            response.headers["X-TEE-Status"] = "error"
            return response

    @staticmethod
    def _extract_user(request: Request) -> str:
        """Extract username from JWT token or headers."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                import base64
                token = auth.split(" ")[1]
                parts = token.split(".")
                if len(parts) >= 2:
                    payload = parts[1]
                    padding = 4 - len(payload) % 4
                    payload += "=" * padding
                    data = json.loads(base64.b64decode(payload))
                    return data.get("sub", data.get("username", "authenticated"))
            except Exception:
                pass
        return request.headers.get("X-User", "anonymous")

    @staticmethod
    def _extract_ip(request: Request) -> str:
        """Extract client IP (supports proxies)."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"