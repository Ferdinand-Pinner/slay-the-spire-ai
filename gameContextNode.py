import lib.slaythespire as slaythespire
from lib.slaythespire import GameContext as CppGameContext  # Assume C++ binding is in `slaythespire`

class GameContextNode:
    def __init__(self, game_context, parent=None):
        if game_context is None:
            raise ValueError("game_context cannot be None")
        self.game_context = game_context
        self.parent = parent
        self.actions = self.available_actions()
        self.children = []

    def create_node(self, game_context):
        """Create a new node in the tree for a given game context."""
        node = {
            'state': game_context,  # This holds the current game context
            'actions': {}  # This will store actions and their q_values
        }
        return node
    
    def clone(self):
        cloned_game_context = self.game_context.clone()
        cloned_wrapper = GameContextNode(cloned_game_context)
        cloned_wrapper.tree = self.tree.copy()  # shallow copy tree if it’s sufficient
        cloned_wrapper.current_node = self.current_node  # if mutable, copy manually
        return cloned_wrapper

    def available_actions(self):        
        if self.game_context is None:
            return []
        
        actions = slaythespire.GameAction.getAllActionsInState(self.game_context)
        if actions is None or len(actions) == 0:
            raise ValueError("No available actions found in the current game context.")
        
        return actions
    
    def perform_action(self, action):
        action.execute(self.game_context)
    
    def is_non_terminal(self): 
        return self.game_context.outcome == slaythespire.GameOutcome.UNDECIDED
    
    def add_action(self, action, q_value):
        """Add an action to the current node with its corresponding Q-value."""
        self.tree[self.current_node]['actions'][action] = {'q_value': q_value}

    def get_action_q_value(self, action):
        """Retrieve the Q-value for a given action in the current node."""
        return self.tree[self.current_node]['actions'].get(action, {}).get('q_value', None)

    def step(self, action):
        """Perform the action and update the game context."""
        self.game_context.step(action)
        self.current_node = self.create_node(self.game_context)  # Update the current node to the new state
    
    def calculate_reward(self):
        """
        Calculate the reward for the current game state.
        Returns:
            float: The reward value.
        """
        # Example reward logic
        outcome = self.game_context.outcome
        if outcome == slaythespire.GameOutcome.PLAYER_VICTORY:
            return 1.0  # Reward for winning
        elif outcome == slaythespire.GameOutcome.PLAYER_LOSS:
            return -1.0  # Penalty for losing
        else:
            # Partial progress reward, e.g., based on health, score, or other metrics
            return self.evaluate_partial_progress()

    def evaluate_partial_progress(self):
        """
        Evaluate the progress of the player to assign a partial reward.
        Returns:
            float: Reward based on intermediate progress.
        """
        # Example: Reward based on remaining health or progress in the game

        max_health = self.game_context.max_hp
        current_health = self.game_context.cur_hp

        if max_health > 0:
            # Reward for percentage of health remaining
            health_reward = current_health / max_health
            # Penalty for health lost compared to the starting point
            health_penalty = (max_health - current_health) / max_health if max_health > current_health else 0
            # Composite score: emphasize reward more than penalty
            return health_reward - 0.5 * health_penalty
        else:
            return -1.0  # Default penalty if health metrics are invalid
    
    @property
    def outcome(self):
        """Access the outcome with additional logic if necessary."""
        return self.game_context.outcome

    # Additional custom methods or helper functions as needed
    # E.g., methods to translate state for ML model input, custom logging, etc.
