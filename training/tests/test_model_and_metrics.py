from __future__ import annotations

import pytest
import torch

from foliage_warden_training.labels import LABEL_TO_INDEX, label_schema_id
from foliage_warden_training.metrics import classification_metrics
from foliage_warden_training.model import ModelConfig, TemporalCnnGru, parameter_count


def test_model_shape_and_size() -> None:
    config = ModelConfig(num_frames=4, image_size=32, feature_dim=16, hidden_dim=24)
    model = TemporalCnnGru(config).eval()

    output = model(torch.zeros(2, 4, 3, 32, 32))

    assert output.shape == (2, 6)
    assert parameter_count(model) < 100_000


def test_model_rejects_wrong_rank() -> None:
    model = TemporalCnnGru(ModelConfig())
    try:
        model(torch.zeros(1, 3, 32, 32))
    except ValueError as error:
        assert "N,T,C,H,W" in str(error)
    else:
        raise AssertionError("wrong-rank input should fail")


def test_model_config_is_bound_to_label_schema() -> None:
    with pytest.raises(ValueError, match="label schema"):
        ModelConfig(num_classes=7)


def test_harmful_binary_metrics_treat_eating_and_digging_as_one_decision() -> None:
    passing = LABEL_TO_INDEX["PASSING"]
    eating = LABEL_TO_INDEX["EATING"]
    digging = LABEL_TO_INDEX["DIGGING"]
    metrics = classification_metrics(
        [eating, digging, passing],
        [digging, digging, eating],
    )

    assert metrics["harmful_binary"]["true_positive"] == 2
    assert metrics["harmful_binary"]["precision"] == 2 / 3
    assert metrics["harmful_binary"]["recall"] == 1.0
    assert len(label_schema_id()) == 64


def test_probability_quality_and_harmful_operating_points() -> None:
    passing = LABEL_TO_INDEX["PASSING"]
    eating = LABEL_TO_INDEX["EATING"]
    probabilities = [
        [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.05, 0.05, 0.7, 0.1, 0.05, 0.05],
    ]

    metrics = classification_metrics(
        [passing, eating],
        [passing, eating],
        probabilities,
    )
    quality = metrics["probability_quality"]

    assert quality["top_2_accuracy"] == 1.0
    assert quality["negative_log_likelihood"] > 0
    assert quality["harmful_probability_operating_points"]["0.50"] == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 1,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
    }
