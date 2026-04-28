"""
routers/oracle.py  — Add to your existing FastAPI backend (Render)

This router is a thin proxy:
  1. Receives the multimodal request from the frontend
  2. Forwards it to the HuggingFace Spaces Oracle service
  3. Streams WebSocket updates back to the client

Add to main.py:
    from routers.oracle import router as oracle_router
    app.include_router(oracle_router)

Set env var on Render:
    ORACLE_SERVICE_URL=https://YOUR-HF-USERNAME-icu-digital-twin-oracle.hf.space
"""

import json
import asyncio
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

from dependencies import get_current_user

log = logging.getLogger("oracle_proxy")

router = APIRouter(prefix="/icu/ai/oracle", tags=["Oracle"])

# ── Set this env var on Render ──────────────────────────────────────────────────
import os
ORACLE_URL = os.getenv("ORACLE_SERVICE_URL", "https://your-hf-space.hf.space")


# ─── Schemas ───────────────────────────────────────────────────────────────────
class OracleRequest(BaseModel):
    patient_id:     str
    image_base64:   str                  # base64 encoded X-ray (with or without data URI prefix)
    clinical_notes: Optional[str] = ""
    vitals:         Optional[dict] = {}  # pass vitals dict directly


# ─── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/assess")
async def start_oracle_assessment(
    req: OracleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Start a multimodal AI assessment.
    Returns a task_id. Connect to WS /icu/ai/oracle/ws/{task_id} for live updates.
    """
    payload = {
        "image_base64":   req.image_base64,
        "clinical_notes": req.clinical_notes,
        "vitals":         req.vitals,
        "patient_id":     req.patient_id,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(f"{ORACLE_URL}/oracle/assess", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, detail=f"Oracle service unreachable at {ORACLE_URL}. Is it deployed?")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, detail=f"Oracle service error: {e.response.text}")


@router.get("/task/{task_id}")
async def poll_oracle_task(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Poll oracle task status (alternative to WebSocket)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{ORACLE_URL}/oracle/task/{task_id}")
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, "Oracle service unreachable")


@router.websocket("/ws/{task_id}")
async def oracle_ws_proxy(websocket: WebSocket, task_id: str):
    """
    WebSocket proxy: connects frontend ↔ Oracle HF Space.
    The frontend connects here; we forward all messages from the Oracle service.
    """
    await websocket.accept()

    # Build WebSocket URL for the HF Space
    oracle_ws_url = ORACLE_URL.replace("https://", "wss://").replace("http://", "ws://")
    oracle_ws_url = f"{oracle_ws_url}/ws/oracle/{task_id}"

    try:
        async with httpx.AsyncClient() as client:
            # Use httpx to poll task status and forward results
            # (Direct WS proxy not available in httpx; we poll and stream)
            max_polls = 120   # 120 * 1s = 2 minutes timeout
            last_progress = -1

            for _ in range(max_polls):
                await asyncio.sleep(1.0)

                try:
                    resp = await client.get(
                        f"{ORACLE_URL}/oracle/task/{task_id}",
                        timeout=5.0
                    )
                    data = resp.json()
                    status = data.get("status", "queued")

                    if status == "complete":
                        await websocket.send_json(data.get("result", data))
                        break
                    elif status == "failed":
                        await websocket.send_json({
                            "type":    "error",
                            "task_id": task_id,
                            "message": data.get("error", "Assessment failed")
                        })
                        break
                    elif status == "processing":
                        # Send a heartbeat so the frontend knows we're alive
                        progress = data.get("progress", 0)
                        if progress != last_progress:
                            await websocket.send_json({
                                "type":     "thinking",
                                "message":  "🔄 Oracle is processing...",
                                "progress": progress,
                                "status":   status,
                            })
                            last_progress = progress
                    else:
                        # queued
                        await websocket.send_json({"type": "thinking", "message": "⏳ Queued...", "progress": 0})

                except (httpx.ConnectError, httpx.TimeoutException):
                    await websocket.send_json({"type": "thinking", "message": "🔄 Waiting for Oracle...", "progress": 0})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WS proxy error for {task_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
