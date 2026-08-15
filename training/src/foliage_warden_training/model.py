from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn

from .labels import BEHAVIOR_LABELS

MODEL_ARCHITECTURE = "temporal-cnn-gru-v1"


@dataclass(frozen=True)
class ModelConfig:
    num_frames: int = 16
    image_size: int = 96
    feature_dim: int = 64
    hidden_dim: int = 96
    gru_layers: int = 1
    dropout: float = 0.1
    num_classes: int = len(BEHAVIOR_LABELS)

    def __post_init__(self) -> None:
        positive = {
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "gru_layers": self.gru_layers,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"model fields must be positive: {', '.join(invalid)}")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_classes != len(BEHAVIOR_LABELS):
            raise ValueError("num_classes must match the fixed behavior label schema")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ModelConfig:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})  # type: ignore[arg-type]


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class TemporalCnnGru(nn.Module):
    """Small frame encoder plus unidirectional GRU for causal clip classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_encoder = nn.Sequential(
            ConvNormAct(3, 16, stride=2),
            SeparableConv(16, 24, stride=2),
            SeparableConv(24, 40, stride=2),
            SeparableConv(40, config.feature_dim, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.temporal = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [N,T,C,H,W], got shape {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        features = self.frame_encoder(frames.reshape(batch * time, channels, height, width))
        sequence = features.reshape(batch, time, -1)
        outputs, _ = self.temporal(sequence)
        return self.classifier(outputs[:, -1])


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    for name, parameter in model.temporal.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "weight_hh" in name:
            nn.init.orthogonal_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)
