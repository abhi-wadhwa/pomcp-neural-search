"""Neural rollout policy for POMCP.

Instead of using a uniform random rollout policy during POMCP simulations,
we train a small MLP to imitate the improved policy discovered by POMCP.

Training procedure (DAgger-style):
1. Run POMCP with random rollouts to collect (belief_features, best_action) pairs
2. Train MLP: belief_features -> action probabilities
3. Use trained MLP as the rollout policy in subsequent POMCP runs
4. Repeat to iteratively improve

The neural policy provides faster, more informed rollouts compared to
random simulation, improving POMCP's sample efficiency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.environments.pomdp_base import POMDPEnv


class RolloutPolicyNetwork(nn.Module):
    """Small MLP for action selection given belief features.

    Architecture: input -> 64 -> ReLU -> 64 -> ReLU -> num_actions (softmax)
    """

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits."""
        return self.network(x)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax over logits)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class NeuralRolloutPolicy:
    """Neural rollout policy wrapper for POMCP.

    Manages training and inference of the rollout policy network.

    Parameters
    ----------
    env : POMDPEnv
        The POMDP environment.
    hidden_dim : int
        Hidden dimension of the MLP.
    learning_rate : float
        Learning rate for training.
    temperature : float
        Softmax temperature for action sampling during rollout.
        Lower = more greedy, higher = more exploratory.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        env: POMDPEnv,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.env = env
        self.device = torch.device(device)
        self.temperature = temperature

        # Determine input dimension from environment
        sample_particles = [env.sample_initial_state() for _ in range(10)]
        features = env.belief_features(sample_particles)
        self.input_dim = len(features)
        self.num_actions = len(env.get_actions())

        self.network = RolloutPolicyNetwork(
            self.input_dim, self.num_actions, hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        # Training data buffer
        self.features_buffer: list[np.ndarray] = []
        self.actions_buffer: list[int] = []

    def add_training_data(
        self, belief_features: np.ndarray, best_action: int
    ) -> None:
        """Add a (belief_features, best_action) training example.

        Called after POMCP selects an action -- we record the belief state
        and the action POMCP chose as a supervised training signal.
        """
        self.features_buffer.append(belief_features.copy())
        self.actions_buffer.append(best_action)

    def train(
        self, epochs: int = 50, batch_size: int = 64, verbose: bool = False
    ) -> dict[str, list[float]]:
        """Train the rollout policy on collected data.

        Returns
        -------
        history : dict
            Training history with 'loss' and 'accuracy' lists.
        """
        if len(self.features_buffer) < batch_size:
            if verbose:
                print(
                    f"Not enough data ({len(self.features_buffer)}) "
                    f"for batch size {batch_size}"
                )
            return {"loss": [], "accuracy": []}

        X = torch.tensor(np.array(self.features_buffer), dtype=torch.float32).to(
            self.device
        )
        y = torch.tensor(self.actions_buffer, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        self.network.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.network(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.shape[0]
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += batch_X.shape[0]

            avg_loss = epoch_loss / total
            accuracy = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"loss={avg_loss:.4f}, acc={accuracy:.3f}"
                )

        return history

    def __call__(self, env: POMDPEnv, state: Any, depth: int) -> Any:
        """Select an action using the neural rollout policy.

        Used as the rollout_policy callback in POMCP.
        Falls back to random if network hasn't been trained.
        """
        return self.select_action(env, state)

    def select_action(self, env: POMDPEnv, state: Any) -> Any:
        """Select action given a single state (used during rollout).

        During rollout, we only have a single state, so we create
        a degenerate particle set and extract features.
        """
        features = env.belief_features([state])
        return self.select_action_from_features(features, env)

    def select_action_from_features(
        self, features: np.ndarray, env: POMDPEnv
    ) -> Any:
        """Select action from precomputed belief features."""
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(
                self.device
            )
            logits = self.network(x).squeeze(0)

            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature

            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Sample action from distribution
        actions = env.get_actions()
        action_idx = np.random.choice(len(actions), p=probs)
        return actions[action_idx]

    def get_action_probs_from_features(self, features: np.ndarray) -> np.ndarray:
        """Get action probability distribution from belief features."""
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(
                self.device
            )
            probs = self.network.get_action_probs(x).squeeze(0).cpu().numpy()
        return probs

    def save(self, path: str) -> None:
        """Save the network weights."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the network weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state_dict"])

    def clear_buffer(self) -> None:
        """Clear the training data buffer."""
        self.features_buffer.clear()
        self.actions_buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Number of training examples in the buffer."""
        return len(self.features_buffer)
