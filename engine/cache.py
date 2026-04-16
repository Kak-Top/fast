# engine/cache.py
from typing import Dict, Optional
import time, hashlib, logging

from config import settings

logger = logging.getLogger(__name__)

class EncryptedKVCache:
    """Store compressed tokens with TTL"""
    
    def __init__(self, ttl_seconds: int = None):
        self._cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
    
    def _generate_key(self, patient_id: str, timestamp: int) -> str:
        window = timestamp // 300  # 5-min windows
        return hashlib.sha256(f"{patient_id}:{window}".encode()).hexdigest()[:20]
    
    def store(self, patient_id: str, encoded_vitals: Dict) -> str:
        key = self._generate_key(patient_id, encoded_vitals['metadata']['timestamp'])
        self._cache[key] = {
            'data': encoded_vitals,
            'expires_at': time.time() + self.ttl,
            'patient_id': patient_id
        }
        return key
    
    def get(self, patient_id: str, max_age_seconds: int = None) -> Optional[Dict]:
        max_age = max_age_seconds or self.ttl
        now = time.time()
        
        for entry in self._cache.values():
            if (entry['patient_id'] == patient_id and 
                entry['expires_at'] > now):
                return entry['data']
        return None