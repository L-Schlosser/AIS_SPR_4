"""Configuration for NER extraction models."""

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for NER token classification training."""

    model_name: str = "distilbert-base-german-cased"
    max_length: int = 256
    batch_size: int = 16
    lr: float = 5e-5
    epochs: int = 20
    weight_decay: float = 0.01
