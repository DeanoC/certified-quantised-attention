"""Calibrated per-layer/profile metadata used by optional benchmark flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibratedProfile:
    epsilon: np.ndarray
    execution_floor: np.ndarray
    v_format: list[str]
    ctx_buckets: list[int]
    model_name: str = ""
    calibration_date: str = ""
    num_prompts: int = 0
    target_cos: float = 0.999

    def _find_bucket(self, ctx_len: int) -> int:
        for i, boundary in enumerate(self.ctx_buckets):
            if ctx_len <= boundary:
                return i
        return len(self.ctx_buckets) - 1

    def get_epsilon(self, ctx_len: int, layer: int, head: int) -> float:
        bucket = self._find_bucket(ctx_len)
        return float(self.epsilon[bucket, layer, head])

    def get_layer_head_epsilons(self, ctx_len: int, layer: int) -> np.ndarray:
        bucket = self._find_bucket(ctx_len)
        return self.epsilon[bucket, layer, :]

    def get_layer_epsilons_min(self, ctx_len: int) -> dict[int, float]:
        bucket = self._find_bucket(ctx_len)
        return {
            layer: float(self.epsilon[bucket, layer, :].min())
            for layer in range(self.epsilon.shape[1])
        }

    def get_layer_epsilons(self, ctx_len: int) -> dict[int, float]:
        return self.get_layer_epsilons_min(ctx_len)

    @classmethod
    def load(cls, path: str) -> "CalibratedProfile":
        data = np.load(path, allow_pickle=True)
        return cls(
            epsilon=data["epsilon"],
            execution_floor=data["execution_floor"],
            v_format=data["v_format"].tolist(),
            ctx_buckets=data["ctx_buckets"].tolist(),
            model_name=str(data["model_name"]) if "model_name" in data else "",
            calibration_date=str(data["calibration_date"]) if "calibration_date" in data else "",
            num_prompts=int(data["num_prompts"]) if "num_prompts" in data else 0,
            target_cos=float(data["target_cos"]) if "target_cos" in data else 0.999,
        )

