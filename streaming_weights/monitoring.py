# streaming_weights/monitoring.py
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StreamingMetrics:
    """Metrics for streaming model performance"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    download_times: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    layer_access_counts: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def avg_download_time(self) -> float:
        return (
            sum(self.download_times) / len(self.download_times)
            if self.download_times
            else 0.0
        )

    @property
    def avg_inference_time(self) -> float:
        return (
            sum(self.inference_times) / len(self.inference_times)
            if self.inference_times
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_download_time_ms": self.avg_download_time * 1000,
            "avg_inference_time_ms": self.avg_inference_time * 1000,
            "most_accessed_layers": sorted(
                self.layer_access_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


class StreamingMonitor:
    """Monitor for streaming model performance"""

    def __init__(self):
        self.metrics = StreamingMetrics()
        self.logger = logging.getLogger(__name__)

    def record_cache_hit(self, layer_idx: int):
        self.metrics.cache_hits += 1
        self.metrics.total_requests += 1
        self.metrics.layer_access_counts[layer_idx] += 1

    def record_cache_miss(self, layer_idx: int, download_time: float):
        self.metrics.cache_misses += 1
        self.metrics.total_requests += 1
        self.metrics.download_times.append(download_time)
        self.metrics.layer_access_counts[layer_idx] += 1

    def record_inference_time(self, inference_time: float):
        self.metrics.inference_times.append(inference_time)

    def log_summary(self):
        metrics_dict = self.metrics.to_dict()
        self.logger.info(f"Streaming Performance Summary: {metrics_dict}")

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()
