"""
Ocean IQ: High-Performance In-Memory LRU Scan Cache.
Designed for low-resource environments (0.5 vCPU, 512 MB RAM).
Caches parsed scan payloads and base64 images to achieve < 5ms response times.
"""

import collections
import threading
import hashlib
import copy
from typing import Optional, Dict, Any


class ScanCache:
    """
    Thread-safe LRU cache with SHA-256 content addressing.
    Ensures repeat scans and built-in sample requests execute in under 5 milliseconds.
    """

    def __init__(self, capacity: int = 40):
        self.capacity = capacity
        self._cache: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def compute_key(raw_bytes: bytes, conf_threshold: float, altitude: Optional[float], draw_tiles: bool) -> str:
        h = hashlib.sha256(raw_bytes).hexdigest()[:16]
        alt_str = f"{altitude:.1f}" if altitude is not None else "none"
        conf_str = f"{conf_threshold:.2f}"
        return f"{h}_c{conf_str}_a{alt_str}_t{int(draw_tiles)}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                # Return deep copy so caller mutations don't alter cached payload
                return copy.deepcopy(self._cache[key])
            return None

    def put(self, key: str, value: Dict[str, Any]):
        with self.lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = copy.deepcopy(value)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def size(self) -> int:
        with self.lock:
            return len(self._cache)

    def clear(self):
        with self.lock:
            self._cache.clear()


# Global singleton instance
scan_cache = ScanCache(capacity=40)
