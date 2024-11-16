import math
import random

class Agent:
    def __init__(self, model, simulations=100):
        self.model = model.PolicyNetwork()  # Initialize the neural network
        self.simulations = simulations
        self.tree = {}

    def mcts_search(self, game_context):
        # Run simulations and select the best action
        for _ in range(self.simulations):
            node = self.simulate(game_context)
            self.backpropagate(node)

        # Select action based on visit count or Q-value
        best_action = max(self.tree[game_context]['actions'], key=lambda a: self.tree[game_context]['actions'][a]['visit_count'])
        return best_action

    def simulate(self, game_context):
        node = game_context.clone()  # Simulate a copy of the game state

        while node.is_non_terminal():
            # If unexplored state, evaluate with neural network
            if node not in self.tree:
                self.expand(node)
                break

            # Use UCB1 for action selection within the tree
            node = self.select_action(node)

        return node

    def expand(self, node):
        state, policy_logits = self.evaluate(node)
        actions = node.available_actions()
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
        # Retrieve the path taken in this simulation
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path
    
    def evaluate(self, node):
        state = self.model.getObservation(node.game_context)  # Get the observation of the game context
        policy_logits = self.model.predict_policy(state)  # This should return action probabilities (before softmax)
        value = self.model.predict_value(state)  # This should return a scalar value for the state
        return value, policy_logits
    
