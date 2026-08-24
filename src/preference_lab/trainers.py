from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import write_metrics
from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str = "outputs"


class PreferenceTrainer:
    """Small CPU trainer used to verify the DPO/ORPO training pipeline."""

    def __init__(self, config: TrainingConfig) -> None:
        if config.method not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")
        self.config = config

    def train(self) -> Path:
        """Run a deterministic CPU training smoke test and save its metric.

        This deliberately avoids downloading model weights. A production experiment can
        replace it with a TRL-backed trainer while retaining the same output contract.
        """
        if self.config.method == "dpo":
            loss = dpo_loss(
                np.array([-0.5, -0.7]),
                np.array([-1.5, -1.1]),
                np.array([-0.6, -0.8]),
                np.array([-1.0, -1.0]),
                self.config.beta,
            )
        elif self.config.method == "orpo":
            loss = orpo_loss(
                np.array([0.5, 0.7]),
                np.array([-0.5, -0.7]),
                np.array([-1.5, -1.1]),
                self.config.lambda_orpo,
            )
        else:
            loss = 0.0
        return write_metrics(
            {"training_loss": loss}, self.config.output_dir, filename="training_metrics.json"
        )
