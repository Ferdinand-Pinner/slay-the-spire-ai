import copy
import lib.slaythespire as slaythespire
from lib.slaythespire import GameContext as CppGameContext  # Assume C++ binding is in `slaythespire`

class GameContextWrapper:
    def __init__(self, game_context):
        self.game_context = game_context
        self.tree = {}  # This will hold the tree structure where each node corresponds to a game state
        self.current_node = self.create_node(game_context)

    def create_node(self, game_context):
        """Create a new node in the tree for a given game context."""
        node = {
            'state': game_context,  # This holds the current game context
            'actions': {}  # This will store actions and their q_values
        }
        return node
    
    def clone(self):
        cloned_game_context = self.game_context.clone()
        cloned_wrapper = GameContextWrapper(cloned_game_context)
        cloned_wrapper.tree = self.tree.copy()  # shallow copy tree if it’s sufficient
        cloned_wrapper.current_node = self.current_node  # if mutable, copy manually
        return cloned_wrapper

    def available_actions(self):
        return CppGameContext.GameAction.getAllActionsInState(self.game_context)
    
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
        
    def is_non_terminal(self):
        # Assuming `self.game_context.outcome` holds the game's outcome state
        return self.game_context.outcome == slaythespire.GameOutcome.UNDECIDED   


    @property
    def outcome(self):
        """Access the outcome with additional logic if necessary."""
        return self.game_context.outcome

    # Additional custom methods or helper functions as needed
    # E.g., methods to translate state for ML model input, custom logging, etc.
