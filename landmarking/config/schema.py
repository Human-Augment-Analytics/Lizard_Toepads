"""Configuration schema for the landmarking package.

Defines nested dataclasses for paths, dataset, model, and training configuration.
Supports JSON serialization and environment variable overrides.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

from .resolver import resolve_path, resolve_dataset_dir


@dataclass
class PathConfig:
    """Filesystem path configuration."""

    data_root: str = ""
    output_root: str = "./runs"
    yolo_obb_model: str = ""


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""

    name: str = "lizard"
    num_landmarks: int = 9
    input_size: int = 512
    graph_topology: str = "chain"
    mean_shape_path: str = ""
    flip_pairs: list = field(default_factory=list)
    data_dir: str = ""
    split_path: str = ""
    landmark_indices: list = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    variant: str = "fused"
    feat_dim: int = 64
    gnn_hidden: int = 128
    num_layers: int = 3
    num_iters: int = 4
    scale_indices: list = field(default_factory=lambda: [0, 1, 2, 3])
    use_coarse_init: bool = True
    heatmap_checkpoint: str = ""
    backbone_pretrained_path: str = ""
    heatmap_size: int = 64
    sigma: float = 1.5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 150
    batch_size: int = 32
    val_batch_size: int = 64
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    lr_milestones: list = field(default_factory=lambda: [60, 90])
    lr_gamma: float = 0.1
    grad_clip: float = 0.5
    init_noise_sigma: float = 0.05
    init_mode: str = "mean"
    rot_factor: float = 0.0
    checkpoint_interval: int = 10
    vis_interval: int = 10
    seed: int = 42
    device: str = "cuda"


@dataclass
class LandmarkingConfig:
    """Top-level configuration for the landmarking package."""

    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str) -> "LandmarkingConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            Populated LandmarkingConfig instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LandmarkingConfig":
        """Construct configuration from a nested dictionary.

        Args:
            data: Dictionary with optional keys: paths, dataset, model, training.

        Returns:
            Populated LandmarkingConfig instance.
        """
        paths = PathConfig(**data.get("paths", {}))
        dataset = DatasetConfig(**data.get("dataset", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        return cls(paths=paths, dataset=dataset, model=model, training=training)

    def resolve_paths(self) -> "LandmarkingConfig":
        """Apply environment variable overrides to path fields.

        Checks for LANDMARKING_{FIELD_NAME} environment variables and
        overrides corresponding fields. Also resolves dataset data_dir
        from data_root if not explicitly set.

        When landmark_indices is non-empty, auto-sets num_landmarks to
        the length of that list.

        Returns:
            self (mutated in place for convenience).
        """
        self.paths.data_root = resolve_path("DATA_ROOT", self.paths.data_root)
        self.paths.output_root = resolve_path("OUTPUT_ROOT", self.paths.output_root)
        self.paths.yolo_obb_model = resolve_path(
            "YOLO_OBB_MODEL", self.paths.yolo_obb_model
        )

        # Resolve dataset data_dir from data_root + dataset name if not set
        if not self.dataset.data_dir:
            self.dataset.data_dir = resolve_dataset_dir(
                self.paths.data_root, self.dataset.name
            )
        else:
            self.dataset.data_dir = resolve_path("DATA_DIR", self.dataset.data_dir)

        # Auto-set num_landmarks from landmark_indices when non-empty
        if self.dataset.landmark_indices:
            self.dataset.num_landmarks = len(self.dataset.landmark_indices)

        return self

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If landmark_indices contains invalid values or duplicates.
        """
        if self.dataset.landmark_indices:
            for idx in self.dataset.landmark_indices:
                if not isinstance(idx, int) or idx < 0 or idx > 97:
                    raise ValueError(
                        f"Invalid landmark index: {idx}. Must be int in [0, 97]."
                    )
            if len(self.dataset.landmark_indices) != len(set(self.dataset.landmark_indices)):
                raise ValueError("landmark_indices contains duplicates")

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Write configuration to a JSON file.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
