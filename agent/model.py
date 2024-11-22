import torch
import torch.nn as nn
import torch.optim as optim
import lib.slaythespire as slaythespire
from typing import Tuple

class PolicyNetwork(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the PolicyNetwork with fully connected layers.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(412, 256)  # Adjust input size as needed
        self.fc2: nn.Linear = nn.Linear(256, 128)
        self.fc3: nn.Linear = nn.Linear(128, 64)
        self.policy_head: nn.Linear = nn.Linear(64, 412)  # Action logits
        self.value_head: nn.Linear = nn.Linear(64, 1)  # Scalar value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor representing the game state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits and value output.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        policy_logits: torch.Tensor = self.policy_head(x)
        value: torch.Tensor = self.value_head(x)
        return policy_logits, value

    def predict_policy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict the policy logits for a given state.

        Args:
            state (torch.Tensor): The state representation as a tensor.

        Returns:
            torch.Tensor: Policy logits representing action probabilities (before softmax).
        """
        policy_logits, _ = self.forward(state)
        return policy_logits

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict the value of a given state.

        Args:
            state (torch.Tensor): The state representation as a tensor.

        Returns:
            torch.Tensor: Scalar value of the state.
        """
        _, value = self.forward(state)
        return value

    def getObservation(self, game_context: slaythespire.GameContext) -> torch.Tensor:
        """
        Get the observation tensor from the game context.

        Args:
            game_context (slaythespire.GameContext): The current game context.

        Returns:
            torch.Tensor: Tensor representation of the game state.
        """
        return slaythespire.NNInterface.getObservation(game_context)
