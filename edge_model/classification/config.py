"""Classification configuration."""

from dataclasses import dataclass, field


@dataclass
class ClassificationConfig:
    """Configuration for document classification model."""

    image_size: int = 224
    num_classes: int = 3
    batch_size: int = 16
    lr_frozen: float = 1e-4
    lr_unfrozen: float = 1e-5
    epochs_frozen: int = 5
    epochs_unfrozen: int = 10
    model_name: str = "tf_efficientnet_lite0"
    class_names: list[str] = field(
        default_factory=lambda: ["arztbesuchsbestaetigung", "lieferschein", "reisekostenbeleg"]
    )
