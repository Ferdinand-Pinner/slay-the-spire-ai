import math
import random

import torch
from agent.model import PolicyNetwork
from gameContextNode import GameContextNode

class Agent:
    def __init__(self, model=None, nn_interface=None, simulations=100):
        self.model = model if model else PolicyNetwork()  # Use provided model or initialize default
        self.nn_interface = nn_interface
        self.simulations = simulations
        self.tree = {}

    def mcts_search(self, gameContextNode):
        # Run simulations and select the best action
        for _ in range(self.simulations):
            node = self.simulate(gameContextNode)
            self.backpropagate(node)

        # Select action based on visit count or Q-value
        best_action = max(self.tree[gameContextNode]['actions'], key=lambda a: self.tree[gameContextNode]['actions'][a]['visit_count'])
        return best_action

    def simulate(self, gameContextNode):

        while gameContextNode.is_non_terminal():
            if gameContextNode not in self.tree:
                self.expand(gameContextNode)
                break

            # Select the next action and assign the resulting node as a child
            next_action = self.select_action(gameContextNode)
            next_game_context = gameContextNode.perform_action(next_action)
            child_node = GameContextNode(next_game_context, parent=gameContextNode)
            gameContextNode.children.append(child_node)
            gameContextNode = child_node

        return gameContextNode


    def expand(self, node):
        state, policy_logits = self.evaluate(node)
        actions = node.actions
        self.tree[node] = {
            'actions': {a: {'visit_count': 0, 'q_value': 0} for a in actions},
            'policy': policy_logits,
            'value': state,
        }

    def select_action(self, node):
        def ucb1(action):
            q = self.tree[node]['actions'][action]['q_value']
            visits = self.tree[node]['actions'][action]['visit_count']
            total_visits = sum(a['visit_count'] for a in self.tree[node]['actions'].values())
            exploration = math.sqrt(math.log(total_visits + 1) / (visits + 1))
            return q + exploration

        return max(self.tree[node]['actions'], key=ucb1)

    def backpropagate(self, node):
        reward = node.calculate_reward()
        path = self.get_path(node)

        for n in path:
            for action in self.tree[n]['actions']:
                self.tree[n]['actions'][action]['visit_count'] += 1
                self.tree[n]['actions'][action]['q_value'] += (reward - self.tree[n]['actions'][action]['q_value']) / \
                                                              self.tree[n]['actions'][action]['visit_count']

    def get_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse to get the path from root to leaf

    
    def evaluate(self, node):
        state = self.nn_interface.getObservation(node.game_context)  # Get the observation of the game context
        state_tensor = torch.tensor(state, dtype=torch.float32)  # Convert state to a tensor
        policy_logits = self.model.predict_policy(state_tensor)  # This should return action probabilities (before softmax)
        value = self.model.predict_value(state_tensor)  # This should return a scalar value for the state
        return value, policy_logits

    
