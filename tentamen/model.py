from typing import Callable, Dict, Protocol

import torch
import torch.nn as nn

Tensor = torch.Tensor


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass


class Linear(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config["input"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["output"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = self.encoder(x)
        return x


class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        """
        yhat is expected to be a vector with d dimensions.
        The highest values in the vector corresponds with
        the correct class.
        """
        return (yhat.argmax(dim=1) == y).sum() / len(yhat)


# TOEVOEGING TBV VRAAG 1D


class gru_model(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.rnn = nn.GRU(
            input_size=config["input"],
            hidden_size=config["hidden"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )

        self.linear = nn.Linear(config["hidden"], config["output"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class gru_attention_model(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.rnn = nn.GRU(
            input_size=config["input"],
            hidden_size=config["hidden"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )

        self.linear = nn.Linear(config["hidden"], config["output"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
