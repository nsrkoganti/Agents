"""Accuracy metrics for surrogate model evaluation."""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from loguru import logger


class AccuracyMetrics:
    """
    Computes R², relative L2 error, max pointwise error, and inference time.
    All the metrics checked against config thresholds.
    """

    def __init__(self, r2_threshold: float = 0.92,
                 rel_l2_max: float = 0.05,
                 max_point_error: float = 0.15):
        self.r2_threshold    = r2_threshold
        self.rel_l2_max      = rel_l2_max
        self.max_point_error = max_point_error

    def compute_all(self, y_pred: np.ndarray,
                    y_true: np.ndarray) -> Dict[str, float]:
        """Compute all accuracy metrics at once."""
        return {
            "r2":              self.r2_score(y_pred, y_true),
            "rel_l2":          self.relative_l2(y_pred, y_true),
            "max_point_error": self.max_pointwise_error(y_pred, y_true),
            "mae":             self.mae(y_pred, y_true),
            "rmse":            self.rmse(y_pred, y_true),
        }

    def r2_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Coefficient of determination R²."""
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-30:
            return 1.0 if ss_res < 1e-30 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def relative_l2(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Relative L2 error: ||pred - true||₂ / ||true||₂."""
        diff_norm = float(np.linalg.norm(y_pred.ravel() - y_true.ravel()))
        true_norm = float(np.linalg.norm(y_true.ravel())) + 1e-30
        return diff_norm / true_norm

    def max_pointwise_error(self, y_pred: np.ndarray,
                             y_true: np.ndarray) -> float:
        """
        Max relative pointwise error: max(|pred - true| / (|true| + eps)).
        Normalized by field range to avoid division by near-zero values.
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        field_range = float(np.max(y_true_flat) - np.min(y_true_flat)) + 1e-30
        pointwise = np.abs(y_pred_flat - y_true_flat) / field_range
        return float(np.max(pointwise))

    def mae(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred.ravel() - y_true.ravel())))

    def rmse(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_pred.ravel() - y_true.ravel()) ** 2)))

    def passes_thresholds(self, metrics: Dict[str, float]) -> bool:
        r2_ok      = metrics.get("r2", 0)           >= self.r2_threshold
        l2_ok      = metrics.get("rel_l2", 1)       <= self.rel_l2_max
        point_ok   = metrics.get("max_point_error", 1) <= self.max_point_error
        return r2_ok and l2_ok and point_ok

    def per_field_metrics(self, y_pred: np.ndarray, y_true: np.ndarray,
                           field_names: list) -> Dict[str, Dict[str, float]]:
        """Compute metrics per output field dimension."""
        results = {}
        n_fields = y_pred.shape[-1] if y_pred.ndim > 1 else 1

        for i, name in enumerate(field_names[:n_fields]):
            if y_pred.ndim > 1:
                pred_i = y_pred[:, i]
                true_i = y_true[:, i]
            else:
                pred_i = y_pred
                true_i = y_true

            results[name] = self.compute_all(pred_i, true_i)

        return results

    def measure_inference_time(self, model, x_sample: np.ndarray,
                                n_runs: int = 10) -> float:
        """Measure average inference time in milliseconds."""
        import torch
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

        try:
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).to(device)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)

            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    model(x_tensor)
                    times.append((time.perf_counter() - t0) * 1000)

            return float(np.median(times))
        except Exception as e:
            logger.warning(f"Inference timing failed: {e}")
            return 999.0
