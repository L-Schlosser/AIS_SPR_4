"""Pipeline configuration for document processing inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the full document processing pipeline."""

    classifier_model_path: str = ""
    extractor_model_paths: dict[str, str] = field(default_factory=dict)
    extractor_tokenizer_paths: dict[str, str] = field(default_factory=dict)
    schemas_dir: str = "data/schemas"
    confidence_threshold: float = 0.7
    use_ocr: bool = True


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        PipelineConfig populated from the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML content is not a valid mapping.
    """
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(data).__name__}")

    return PipelineConfig(
        classifier_model_path=data.get("classifier_model_path", ""),
        extractor_model_paths=data.get("extractor_model_paths", {}),
        extractor_tokenizer_paths=data.get("extractor_tokenizer_paths", {}),
        schemas_dir=data.get("schemas_dir", "data/schemas"),
        confidence_threshold=data.get("confidence_threshold", 0.7),
        use_ocr=data.get("use_ocr", True),
    )
