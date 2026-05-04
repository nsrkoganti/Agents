"""Normalizes coordinates and field data for model training."""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NormStats:
    coord_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coord_std:  np.ndarray = field(default_factory=lambda: np.ones(3))
    field_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class DataNormalizer:
    """
    Zero-mean, unit-variance normalization for coordinates and fields.
    Stores stats for inverse-transform at inference time.
    """

    def __init__(self):
        self.stats: Optional[NormStats] = None

    def fit_transform(self, coords: np.ndarray,
                      fields: Dict[str, np.ndarray]
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.stats = NormStats()

        self.stats.coord_mean = coords.mean(axis=0)
        self.stats.coord_std  = coords.std(axis=0) + 1e-8
        norm_coords = (coords - self.stats.coord_mean) / self.stats.coord_std

        norm_fields = {}
        for name, arr in fields.items():
            mean = float(np.nanmean(arr))
            std  = float(np.nanstd(arr)) + 1e-8
            self.stats.field_stats[name] = (mean, std)
            norm_fields[name] = (arr - mean) / std

        return norm_coords, norm_fields

    def transform(self, coords: np.ndarray,
                  fields: Dict[str, np.ndarray]
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self.stats is None:
            raise RuntimeError("Call fit_transform first")
        norm_coords = (coords - self.stats.coord_mean) / self.stats.coord_std
        norm_fields = {}
        for name, arr in fields.items():
            if name in self.stats.field_stats:
                mean, std = self.stats.field_stats[name]
                norm_fields[name] = (arr - mean) / std
            else:
                norm_fields[name] = arr
        return norm_coords, norm_fields

    def inverse_transform_fields(self,
                                  fields: Dict[str, np.ndarray]
                                  ) -> Dict[str, np.ndarray]:
        if self.stats is None:
            return fields
        result = {}
        for name, arr in fields.items():
            if name in self.stats.field_stats:
                mean, std = self.stats.field_stats[name]
                result[name] = arr * std + mean
            else:
                result[name] = arr
        return result
