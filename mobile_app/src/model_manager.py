"""Model manager for scanning and reporting on ONNX model files."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a single model file."""

    name: str
    path: str
    size_mb: float
    exists: bool


class ModelManager:
    """Manages ONNX model files for the edge-AI pipeline.

    Scans a directory for ONNX models and reports on their
    availability and sizes.
    """

    REQUIRED_MODELS = [
        ("classifier", "classification/models/classifier_int8.onnx"),
        ("extractor_arztbesuch", "extraction/models/arztbesuch/model.onnx"),
        ("extractor_reisekosten", "extraction/models/reisekosten/model.onnx"),
        ("extractor_lieferschein", "extraction/models/lieferschein/model.onnx"),
    ]

    def __init__(self, models_dir: str) -> None:
        """Initialize with the base models directory.

        Args:
            models_dir: Root directory containing model subdirectories
                        (e.g., ``edge_model``).
        """
        self._models_dir = models_dir

    def get_model_info(self) -> dict[str, ModelInfo]:
        """Return information about all required models.

        Returns:
            Dict mapping model name to its ModelInfo.
        """
        info: dict[str, ModelInfo] = {}
        for name, rel_path in self.REQUIRED_MODELS:
            full_path = os.path.join(self._models_dir, rel_path)
            exists = os.path.isfile(full_path)
            size_mb = 0.0
            if exists:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
            info[name] = ModelInfo(name=name, path=full_path, size_mb=round(size_mb, 2), exists=exists)
        return info

    def check_models_exist(self) -> tuple[bool, list[str]]:
        """Check whether all required models are present.

        Returns:
            Tuple of (all_present, list_of_missing_model_names).
        """
        info = self.get_model_info()
        missing = [mi.name for mi in info.values() if not mi.exists]
        return (len(missing) == 0, missing)

    def get_total_size_mb(self) -> float:
        """Return the total size of all existing models in MB.

        Returns:
            Sum of sizes for models that exist on disk.
        """
        info = self.get_model_info()
        return round(sum(mi.size_mb for mi in info.values() if mi.exists), 2)
