"""PyTorch components used by the COMP 472 neural-network experiments."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ModelConfig:
    """All tunable model and training choices for a single experiment."""

    hidden_sizes: tuple[int, ...] = (256, 64)
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 128
    momentum: float = 0.9
    dropout: float = 0.0
    weight_decay: float = 0.0
    early_stopping: bool = False
    patience: int = 6
    activation: str = "relu"

    def description(self) -> str:
        """Return a stable, readable config string for CSV outputs."""
        values = asdict(self)
        values["hidden_sizes"] = list(self.hidden_sizes)
        return "; ".join(f"{key}={value}" for key, value in values.items())


class MLP(nn.Module):
    """Feed-forward classifier with optional dropout."""

    def __init__(
        self, input_size: int, output_size: int, config: ModelConfig
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        activation_functions = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        if config.activation not in activation_functions:
            raise ValueError(f"Unknown activation {config.activation!r}")
        activation = activation_functions[config.activation]
        previous_size = input_size
        for width in config.hidden_sizes:
            layers.append(nn.Linear(previous_size, width))
            layers.append(activation())
            if config.dropout:
                # Dropout discourages reliance on any one hidden activation.
                layers.append(nn.Dropout(config.dropout))
            previous_size = width
        layers.append(nn.Linear(previous_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


@dataclass
class TrainingResult:
    """The selected model and epoch-by-epoch metrics from one training run."""

    model: MLP
    history: list[dict[str, float | int]]
    best_epoch: int
    best_accuracy: float
    best_loss: float


def set_seed(seed: int) -> None:
    """Set all requested RNGs so data order and model initialization repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _loader(
    features: np.ndarray,
    labels: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    if labels is None:
        dataset = TensorDataset(feature_tensor)
    else:
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        dataset = TensorDataset(feature_tensor, label_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def evaluate(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Calculate sample-weighted loss and accuracy in evaluation mode."""
    loader = _loader(features, labels, batch_size, shuffle=False, seed=0)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            total_loss += criterion(logits, batch_labels).item() * len(batch_labels)
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += len(batch_labels)
    return total_loss / total, correct / total


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    output_size: int,
    config: ModelConfig,
    epochs: int,
    seed: int,
    device: torch.device,
    validation_features: Optional[np.ndarray] = None,
    validation_labels: Optional[np.ndarray] = None,
    early_stopping: bool = False,
    restore_best_checkpoint: bool = True,
) -> TrainingResult:
    """Train a model, selecting the best-accuracy checkpoint on validation data.

    When requested, early stopping monitors validation loss, while checkpoint
    selection remains based on validation accuracy for consistency across versions.
    """
    set_seed(seed)
    model = MLP(features.shape[1], output_size, config).to(device)
    loader = _loader(features, labels, config.batch_size, shuffle=True, seed=seed)
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            # Weight decay discourages unnecessarily large weights and can reduce overfit.
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}")

    has_validation = validation_features is not None and validation_labels is not None
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_val_loss_for_stopping = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_labels)
            train_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            train_count += len(batch_labels)

        train_loss = train_loss_sum / train_count
        train_accuracy = train_correct / train_count
        if has_validation:
            val_loss, val_accuracy = evaluate(
                model,
                validation_features,
                validation_labels,
                config.batch_size,
                device,
                criterion,
            )
        else:
            val_loss, val_accuracy = float("nan"), float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if has_validation:
            if val_accuracy > best_accuracy or (
                val_accuracy == best_accuracy and val_loss < best_loss
            ):
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if early_stopping:
                if val_loss < best_val_loss_for_stopping:
                    best_val_loss_for_stopping = val_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            if early_stopping and stale_epochs >= config.patience:
                break

    if has_validation and restore_best_checkpoint:
        model.load_state_dict(best_state)
    elif not has_validation:
        best_epoch = epochs
    return TrainingResult(model, history, best_epoch, best_accuracy, best_loss)


def predict_probabilities(
    model: MLP,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return row-ordered softmax probabilities for an array of features."""
    loader = _loader(features, None, batch_size, shuffle=False, seed=0)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def one_factor_candidates(
    version: int, base: ModelConfig
) -> list[tuple[str, str, ModelConfig]]:
    """Describe version-specific sweeps, changing only one field at a time."""
    if version == 1:
        return [("baseline", "fixed", base)]
    if version == 2:
        return (
            [("learning_rate", str(value), replace(base, learning_rate=value)) for value in (1e-2, 1e-3, 1e-4)]
            + [("batch_size", str(value), replace(base, batch_size=value)) for value in (64, 128, 256)]
        )
    if version == 3:
        return (
            [("dropout", str(value), replace(base, dropout=value)) for value in (0.0, 0.2, 0.5)]
            + [("weight_decay", str(value), replace(base, weight_decay=value)) for value in (0.0, 1e-4, 1e-3)]
        )
    if version == 4:
        candidates: list[tuple[str, str, ModelConfig]] = []
        candidates.extend(
            (
                "width",
                str(width),
                replace(base, hidden_sizes=(width, width)),
            )
            for width in (128, 256, 512, 1024)
        )
        candidates.extend(
            (
                "hidden_layers",
                str(depth),
                replace(base, hidden_sizes=(256,) * depth),
            )
            for depth in (1, 2, 3)
        )
        return candidates
    raise ValueError(f"Version {version} does not have a hyperparameter sweep")


def config_key(config: ModelConfig) -> tuple[object, ...]:
    """Produce a hashable key for reusing duplicate baseline sweep runs."""
    return tuple(asdict(config).items())
