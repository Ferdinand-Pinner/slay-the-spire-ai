import math
import random
import lib.slaythespire as slaythespire
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from agent.model import PolicyNetwork
from gameContextNode import GameContextNode

class Agent:
    def __init__(self, model: Optional[PolicyNetwork] = None, nn_interface: slaythespire.NNInterface = None, simulations: int = 100) -> None:
        """
        Initialize the Agent.

        Args:
            model (Optional[PolicyNetwork]): Policy network for value and policy prediction.
            nn_interface (Any): Neural network interface for interacting with the game context.
            simulations (int): Number of Monte Carlo Tree Search simulations to run per move.
        """
        self.model: PolicyNetwork = model if model else PolicyNetwork()  # Use provided model or initialize default
        self.nn_interface: slaythespire.NNInterface = nn_interface
        self.simulations: int = simulations
        self.tree: Dict[GameContextNode, Dict[str, Any]] = {}

    def mcts_search(self, gameContextNode: GameContextNode) -> Any:
        """
        Perform MCTS to find the best action from the current game context.

        Args:
            gameContextNode (GameContextNode): The root node for MCTS.

        Returns:
            Any: The selected action.
        """
        for _ in range(self.simulations):
            node = self.simulate(gameContextNode)
            self.backpropagate(node)

        # Select action based on visit count or Q-value
        best_action = max(
            self.tree[gameContextNode]['actions'],
            key=lambda a: self.tree[gameContextNode]['actions'][a]['visit_count']
        )
        return best_action

    def simulate(self, gameContextNode: GameContextNode) -> GameContextNode:
        """
        Simulate a rollout in the game tree.

        Args:
            gameContextNode (GameContextNode): The starting node for the simulation.

        Returns:
            GameContextNode: The leaf node reached during the simulation.
        """
        while gameContextNode.is_non_terminal():
            if gameContextNode not in self.tree:
                self.expand(gameContextNode)
                break

            # Select the next action and assign the resulting node as a child
            next_action = self.select_action(gameContextNode)
            next_game_context = gameContextNode.clone()
            next_game_context.perform_action(next_action)
            child_node = GameContextNode(next_game_context.game_context, parent=gameContextNode)
            gameContextNode.children.append(child_node)
            gameContextNode = child_node

        return gameContextNode

    def expand(self, node: GameContextNode) -> None:
        """
        Expand the current node in the game tree.

        Args:
            node (GameContextNode): The node to expand.
        """
        state, policy_logits = self.evaluate(node)
        actions = node.actions
        self.tree[node] = {
            'actions': {a: {'visit_count': 0, 'q_value': 0} for a in actions},
            'policy': policy_logits,
            'value': state,
        }

    def select_action(self, node: GameContextNode) -> Any:
        """
        Select the best action from the current node using UCB1.

        Args:
            node (GameContextNode): The node to select an action from.

        Returns:
            Any: The selected action.
        """
        def ucb1(action: Any) -> float:
            q = self.tree[node]['actions'][action]['q_value']
            visits = self.tree[node]['actions'][action]['visit_count']
            total_visits = sum(a['visit_count'] for a in self.tree[node]['actions'].values())
            exploration = math.sqrt(math.log(total_visits + 1) / (visits + 1))
            return q + exploration

        return max(self.tree[node]['actions'], key=ucb1)

    def backpropagate(self, node: GameContextNode) -> None:
        """
        Backpropagate the reward through the path from the leaf node to the root.

        Args:
            node (GameContextNode): The leaf node reached during simulation.
        """
        reward = node.calculate_reward()
        path = self.get_path(node)

        for n in path:
            for action in self.tree[n]['actions']:
                visit_count = self.tree[n]['actions'][action]['visit_count']
                q_value = self.tree[n]['actions'][action]['q_value']
                self.tree[n]['actions'][action]['visit_count'] += 1
                self.tree[n]['actions'][action]['q_value'] += (reward - q_value) / (visit_count + 1)

    def get_path(self, node: GameContextNode) -> List[GameContextNode]:
        """
        Get the path from the root to the given node.

        Args:
            node (GameContextNode): The target node.

        Returns:
            List[GameContextNode]: The path from the root to the node.
        """
        path: List[GameContextNode] = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse to get the path from root to leaf

    def evaluate(self, node: GameContextNode) -> Tuple[float, torch.Tensor]:
        """
        Evaluate the value and policy for the given node.

        Args:
            node (GameContextNode): The node to evaluate.

        Returns:
            Tuple[float, torch.Tensor]: The value and policy logits for the state.
        """
        state: List[float] = self.nn_interface.getObservation(node.game_context)
        if not state:
            raise ValueError("Observation returned an empty state.")

        state_tensor = torch.tensor(state, dtype=torch.float32)
        policy_logits = self.model.predict_policy(state_tensor)
        value = self.model.predict_value(state_tensor)

        return value.item(), policy_logits
