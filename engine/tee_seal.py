# engine/tee_seal.py
"""
TEE Sealing + Merkle Audit Trail
──────────────────────────────────
Provides:
  • seal(data)          → HMAC-SHA256 proof  (tamper detection)
  • verify(data, proof) → bool
  • MerkleAuditTrail   → append-only, ZKP-style membership proofs

This is a *software* simulation of hardware TEE sealing.
In production with real Intel SGX the ENCLAVE_MASTER_KEY would be
a hardware-bound secret that never leaves the enclave.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Master key (read from env, fallback to dev default) ───────────────────────
_MASTER_KEY: bytes = os.getenv(
    "ENCLAVE_MASTER_KEY", "ICU_DIGITAL_TWIN_TEE_KEY_32BYTES!"
).encode()[:32].ljust(32, b"0")


# ── HMAC Sealing ─────────────────────────────────────────────────────────────

def seal(data: Any) -> str:
    """Return hex HMAC-SHA256 proof for arbitrary JSON-serialisable data."""
    payload = json.dumps(data, sort_keys=True, default=str).encode()
    return hmac.new(_MASTER_KEY, payload, hashlib.sha256).hexdigest()


def verify(data: Any, proof: str) -> bool:
    """Return True if proof matches data."""
    expected = seal(data)
    return hmac.compare_digest(expected, proof)


def sealed_response(data: Dict) -> Dict:
    """Wrap any response dict with a TEE seal + timestamp."""
    proof = seal(data)
    return {
        **data,
        "_tee": {
            "sealed": True,
            "proof": proof,
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "algorithm": "HMAC-SHA256",
        },
    }


# ── Merkle Audit Trail ────────────────────────────────────────────────────────

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    if not leaves:
        return _sha256("empty")
    layer = [_sha256(leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last if odd
        layer = [
            _sha256(layer[i] + layer[i + 1])
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


class MerkleAuditTrail:
    """
    Append-only Merkle tree audit log.

    Every entry is hashed. The Merkle root changes with every append —
    any tampering is immediately detectable by comparing roots.
    """

    def __init__(self):
        self._entries: List[Dict] = []
        self._leaf_hashes: List[str] = []
        self._root: str = _sha256("empty")

    def append(self, event_type: str, actor: str, data: Dict) -> Dict:
        entry = {
            "index": len(self._entries),
            "event_type": event_type,
            "actor": actor,
            "data": data,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        leaf_str = json.dumps(entry, sort_keys=True, default=str)
        leaf_hash = _sha256(leaf_str)

        self._entries.append(entry)
        self._leaf_hashes.append(leaf_hash)
        self._root = _merkle_root(self._leaf_hashes)

        return {"index": entry["index"], "leaf_hash": leaf_hash, "root": self._root}

    def root(self) -> str:
        return self._root

    def recent(self, n: int = 10) -> List[Dict]:
        return self._entries[-n:]

    def verify_integrity(self) -> Dict:
        """Recompute root from all leaves and compare."""
        recomputed = _merkle_root(self._leaf_hashes)
        ok = recomputed == self._root
        return {
            "integrity_ok": ok,
            "stored_root": self._root,
            "recomputed_root": recomputed,
            "total_entries": len(self._entries),
        }

    def generate_proof(self, index: int) -> Optional[Dict]:
        """Generate Merkle inclusion proof for entry at index (ZKP-style)."""
        if index < 0 or index >= len(self._leaf_hashes):
            return None

        leaf_hash = self._leaf_hashes[index]
        proof_path = []
        layer = list(self._leaf_hashes)
        idx = index

        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            sibling_idx = idx ^ 1  # XOR flips last bit → sibling
            sibling_hash = layer[sibling_idx]
            direction = "right" if idx % 2 == 0 else "left"
            proof_path.append({"hash": sibling_hash, "direction": direction})

            # Move up
            layer = [
                _sha256(layer[i] + layer[i + 1])
                for i in range(0, len(layer), 2)
            ]
            idx //= 2

        return {
            "leaf_hash": leaf_hash,
            "proof": proof_path,
            "root_hash": self._root,
        }

    def verify_proof(self, leaf_hash: str, proof: List[Dict], root_hash: str) -> bool:
        """Verify a Merkle inclusion proof."""
        current = _sha256(leaf_hash)
        for step in proof:
            sibling = step["hash"]
            if step["direction"] == "right":
                current = _sha256(current + sibling)
            else:
                current = _sha256(sibling + current)
        return current == root_hash

    def __len__(self):
        return len(self._entries)


# ── Global Merkle trail instance ──────────────────────────────────────────────
audit_trail = MerkleAuditTrail()
