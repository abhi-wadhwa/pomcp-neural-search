"""Neural value network for POMCP leaf evaluation.

Instead of using Monte Carlo rollouts to evaluate leaf nodes in the
POMCP search tree, we train a value network V(b) that predicts the
expected return from a belief state b.

Training procedure:
1. Run POMCP episodes, collecting (belief_features, discounted_return) pairs
2. Train MLP: belief_features -> scalar value
3. Use V(b) as leaf evaluator in subsequent POMCP runs

This is analogous to the value network in AlphaGo, but operating on
belief features rather than raw observations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.environments.pomdp_base import POMDPEnv


class ValueNetwork(nn.Module):
    """MLP that predicts value from belief features.

    Architecture: input -> 128 -> ReLU -> 64 -> ReLU -> 1
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar value predictions."""
        return self.network(x).squeeze(-1)


class NeuralValueNetwork:
    """Neural value estimator wrapper for POMCP.

    Manages training and inference of the value network, which
    replaces random rollouts at leaf nodes.

    Parameters
    ----------
    env : POMDPEnv
        The POMDP environment.
    hidden_dim : int
        Hidden dimension of the MLP.
    learning_rate : float
        Learning rate for training.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        env: POMDPEnv,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.env = env
        self.device = torch.device(device)

        # Determine input dimension
        sample_particles = [env.sample_initial_state() for _ in range(10)]
        features = env.belief_features(sample_particles)
        self.input_dim = len(features)

        self.network = ValueNetwork(self.input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Normalize targets
        r_min, r_max = env.get_reward_range()
        self.value_scale = max(abs(r_min), abs(r_max), 1.0)

        # Training data buffer
        self.features_buffer: list[np.ndarray] = []
        self.values_buffer: list[float] = []

    def add_training_data(
        self, belief_features: np.ndarray, discounted_return: float
    ) -> None:
        """Add a (belief_features, return) training example."""
        self.features_buffer.append(belief_features.copy())
        self.values_buffer.append(discounted_return / self.value_scale)

    def train(
        self, epochs: int = 100, batch_size: int = 64, verbose: bool = False
    ) -> dict[str, list[float]]:
        """Train the value network on collected data.

        Returns
        -------
        history : dict
            Training history with 'loss' key.
        """
        if len(self.features_buffer) < batch_size:
            if verbose:
                print(f"Not enough data ({len(self.features_buffer)})")
            return {"loss": []}

        X = torch.tensor(np.array(self.features_buffer), dtype=torch.float32).to(
            self.device
        )
        y = torch.tensor(self.values_buffer, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: dict[str, list[float]] = {"loss": []}
        self.network.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            total = 0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                predictions = self.network(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.shape[0]
                total += batch_X.shape[0]

            avg_loss = epoch_loss / total
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        return history

    def __call__(self, env: POMDPEnv, particles: list) -> float:
        """Estimate value from a particle set.

        Used as the value_estimator callback in POMCP.
        """
        return self.predict(env, particles)

    def predict(self, env: POMDPEnv, particles: list) -> float:
        """Predict value from particles."""
        features = env.belief_features(particles)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> float:
        """Predict value from precomputed belief features."""
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(
                self.device
            )
            value = self.network(x).item()
        return value * self.value_scale

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Predict values for a batch of belief features."""
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
            values = self.network(x).cpu().numpy()
        return values * self.value_scale

    def save(self, path: str) -> None:
        """Save the network weights."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "input_dim": self.input_dim,
                "value_scale": self.value_scale,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the network weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.value_scale = checkpoint.get("value_scale", self.value_scale)

    def clear_buffer(self) -> None:
        """Clear the training data buffer."""
        self.features_buffer.clear()
        self.values_buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Number of training examples in the buffer."""
        return len(self.features_buffer)
