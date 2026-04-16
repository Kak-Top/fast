# engine/turbo_quant.py
import numpy as np
import torch
from typing import Union, Dict, List, Optional
import hashlib, time, logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

class PolarQuantEncoder:
    """
    TurboQuant: 32-bit vitals → 3-bit Polar tokens
    """
    
    def __init__(self, input_dim: int = None, seed: int = 42):
        self.input_dim = input_dim or settings.POLAR_INPUT_DIM
        self.quant_levels = 2 ** settings.QUANT_BITS  # 8 levels for 3-bit
        
        # Generate orthogonal rotation matrix M (simulated SGX enclave)
        self._init_rotation_matrix(seed)
        
        # Pre-compute theta bins
        self.theta_bins = np.linspace(0, 2*np.pi, self.quant_levels + 1)[:-1]
        
        logger.info(f"✅ PolarQuantEncoder initialized: {self.input_dim}D → {settings.QUANT_BITS}-bit")
    
    def _init_rotation_matrix(self, seed: int):
        """Generate protected orthogonal matrix M"""
        rng = np.random.default_rng(seed)
        random_matrix = rng.standard_normal((self.input_dim, self.input_dim))
        self.M, _ = np.linalg.qr(random_matrix)  # QR = orthogonal
        self.M = torch.tensor(self.M, dtype=torch.float32)
        self.matrix_hash = hashlib.sha256(self.M.numpy().tobytes()).hexdigest()[:16]
    
    def _cartesian_to_polar(self, x: torch.Tensor) -> tuple:
        """Convert 2D sub-vectors to polar coordinates (r, θ)"""
        if x.shape[-1] % 2 == 1:
            x = torch.cat([x, torch.zeros(1)], dim=-1)
        
        x_reshaped = x.view(-1, 2)
        r = torch.norm(x_reshaped, dim=1)
        theta = torch.atan2(x_reshaped[:, 1], x_reshaped[:, 0]) % (2 * np.pi)
        return r, theta
    
    def _quantize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Map polar angle θ to 3-bit integer (0-7)"""
        quantized = torch.bucketize(theta, torch.tensor(self.theta_bins))
        return quantized.to(torch.uint8)
    
    def encode(self, vitals: Union[np.ndarray, List[float], torch.Tensor], 
               patient_id: Optional[str] = None) -> Dict:
        """
        Main encoding: Raw vitals → 3-bit bitstream
        """
        start_time = time.time()
        
        # Normalize to [0, 1]
        x_raw = torch.tensor(vitals, dtype=torch.float32) if not isinstance(vitals, torch.Tensor) else vitals
        x_min, x_max = x_raw.min(), x_raw.max()
        x_norm = (x_raw - x_min) / (x_max - x_min + 1e-8)
        
        # Step 1: Random orthogonal rotation
        x_rotated = x_norm @ self.M
        
        # Step 2: Polar transformation
        radii, angles = self._cartesian_to_polar(x_rotated)
        
        # Step 3: Quantize to 3-bit
        theta_3bit = self._quantize_theta(angles)
        
        # Compute stats
        latency_ms = (time.time() - start_time) * 1000
        original_bytes = x_raw.numel() * 4
        compressed_bytes = len(theta_3bit.numpy().tobytes())
        
        return {
            'bitstream': theta_3bit.numpy().tobytes(),
            'metadata': {
                'patient_id': patient_id,
                'timestamp': int(time.time()),
                'rotation_matrix_id': self.matrix_hash,
                'encoding_latency_ms': round(latency_ms, 2)
            },
            'stats': {
                'original_size_bytes': original_bytes,
                'compressed_size_bytes': compressed_bytes,
                'compression_ratio': f"{original_bytes/compressed_bytes:.1f}x",
                'vram_saved_percent': round((1 - compressed_bytes/original_bytes) * 100, 1)
            }
        }