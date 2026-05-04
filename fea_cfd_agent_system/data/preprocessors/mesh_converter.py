"""Converts between mesh representations (unstructured ↔ structured grid)."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from loguru import logger


class MeshConverter:
    """
    Adapts mesh data to the format expected by different model families.
    - GNN models need edge_index + edge_features
    - FNO models need structured grids
    - Transolver/PINN accept raw point clouds (batch, N, D)
    """

    def to_point_cloud(self, data: Dict[str, Any],
                        output_fields: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, Y) tensors ready for training:
        X: (N, input_dim) — coordinates [+ other input features]
        Y: (N, output_dim) — target field values
        """
        coords = data["coordinates"]  # (N, 3)
        fields = data["fields"]

        target_arrays = []
        for field_name in output_fields:
            if field_name in fields:
                arr = fields[field_name]
                if arr.ndim == 1:
                    target_arrays.append(arr[:, None])
                else:
                    target_arrays.append(arr)
            else:
                logger.warning(f"Output field '{field_name}' not found in data")

        if not target_arrays:
            raise ValueError(f"None of {output_fields} found in data fields: "
                             f"{list(fields.keys())}")

        Y = np.concatenate(target_arrays, axis=-1).astype(np.float32)
        X = coords.astype(np.float32)

        return X, Y

    def to_graph(self, data: Dict[str, Any],
                  output_fields: list,
                  k_neighbors: int = 8) -> Dict[str, Any]:
        """Build k-NN graph for GNN models."""
        X, Y = self.to_point_cloud(data, output_fields)
        edge_index, edge_features = self._build_knn_graph(X, k_neighbors)
        return {
            "node_features": X,
            "edge_index":    edge_index,
            "edge_features": edge_features,
            "targets":       Y,
        }

    def _build_knn_graph(self, coords: np.ndarray,
                          k: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
            nn.fit(coords)
            distances, indices = nn.kneighbors(coords)

            N = len(coords)
            src_list, dst_list = [], []
            for i in range(N):
                for j_idx in range(1, k + 1):
                    j = indices[i, j_idx]
                    src_list.append(i)
                    dst_list.append(j)

            edge_index = np.stack([src_list, dst_list], axis=0)
            diffs = coords[src_list] - coords[dst_list]
            dists = np.linalg.norm(diffs, axis=-1, keepdims=True)
            edge_features = np.concatenate([diffs, dists], axis=-1).astype(np.float32)
            return edge_index, edge_features

        except ImportError:
            logger.warning("sklearn not available — returning empty graph")
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    def to_structured_grid(self, data: Dict[str, Any],
                            output_fields: list,
                            grid_res: int = 64) -> Optional[Tuple]:
        """
        Interpolate unstructured data onto a structured grid for FNO.
        Returns (X_grid, Y_grid) of shape (C, H, W).
        """
        coords = data["coordinates"]
        fields = data["fields"]

        try:
            from scipy.interpolate import griddata
        except ImportError:
            logger.warning("scipy not available — cannot convert to structured grid")
            return None

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(y_min, y_max, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        grid_fields = []
        for field_name in output_fields:
            if field_name in fields:
                arr = fields[field_name]
                if arr.ndim > 1:
                    arr = np.linalg.norm(arr, axis=-1)
                gi = griddata(coords[:, :2], arr, (Xi, Yi),
                              method="linear", fill_value=float(np.nanmean(arr)))
                grid_fields.append(gi)

        if not grid_fields:
            return None

        Y_grid = np.stack(grid_fields, axis=0).astype(np.float32)
        coord_grid = np.stack([Xi, Yi], axis=0).astype(np.float32)
        return coord_grid, Y_grid
