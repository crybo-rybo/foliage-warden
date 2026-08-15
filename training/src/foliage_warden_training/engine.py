from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .labels import BEHAVIOR_LABELS
from .metrics import classification_metrics
from .model import TemporalCnnGru


def evaluate_model(
    model: TemporalCnnGru,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
    include_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    all_probabilities: list[list[float]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    sample_count = 0

    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            logits = model(frames)
            probabilities = torch.softmax(logits, dim=1)
            predicted = probabilities.argmax(dim=1)
            batch_size = labels.shape[0]
            if criterion is not None:
                total_loss += float(criterion(logits, labels).item()) * batch_size
            sample_count += batch_size

            batch_targets = labels.cpu().tolist()
            batch_predictions = predicted.cpu().tolist()
            batch_probabilities = probabilities.cpu().tolist()
            targets.extend(batch_targets)
            predictions.extend(batch_predictions)
            all_probabilities.extend(batch_probabilities)

            if include_predictions:
                for clip_id, target, prediction, probability_values in zip(
                    batch["clip_id"],
                    batch_targets,
                    batch_predictions,
                    batch_probabilities,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "clip_id": clip_id,
                            "actual": BEHAVIOR_LABELS[target],
                            "predicted": BEHAVIOR_LABELS[prediction],
                            "confidence": probability_values[prediction],
                            "probabilities": dict(
                                zip(BEHAVIOR_LABELS, probability_values, strict=True)
                            ),
                        }
                    )

    metrics = classification_metrics(targets, predictions, all_probabilities)
    if criterion is not None:
        metrics["loss"] = total_loss / sample_count
    return metrics, prediction_rows
