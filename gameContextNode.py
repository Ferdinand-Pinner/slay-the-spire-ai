from typing import Optional, List
import lib.slaythespire as slaythespire

class GameContextNode:
    def __init__(self, game_context: slaythespire.GameContext, parent: Optional['GameContextNode'] = None):
        if not game_context:
            raise ValueError("game_context cannot be None")
        self.game_context: slaythespire.GameContext = game_context
        self.parent: Optional['GameContextNode'] = parent
        self.actions: List[slaythespire.GameAction] = self.available_actions()
        self.children: List['GameContextNode'] = []

    def clone(self) -> 'GameContextNode':
        """Clone the current node and its game context."""
        cloned_game_context = self.game_context.clone()
        return GameContextNode(cloned_game_context, parent=self)

    def available_actions(self) -> List[slaythespire.GameAction]:
        """Retrieve all available actions in the current game state."""
        if self.game_context is None:
            return []
        actions = slaythespire.GameAction.getAllActionsInState(self.game_context)
        return actions if actions else []  # Return empty list if no actions

    def perform_action(self, action: slaythespire.GameAction) -> None:
        """Execute the given action in the current game context."""
        if action is None:
            raise ValueError("Action cannot be None")
        action.execute(self.game_context)

    def is_non_terminal(self) -> bool:
        """Check if the game is still undecided."""
        return self.game_context.outcome == slaythespire.GameOutcome.UNDECIDED

    def calculate_reward(self) -> float:
        """
        Calculate the reward for the current game state.
        Returns:
            float: The reward value.
        """
        outcome = self.game_context.outcome
        if outcome == slaythespire.GameOutcome.PLAYER_VICTORY:
            return 1.0
        elif outcome == slaythespire.GameOutcome.PLAYER_LOSS:
            return -1.0
        return self.evaluate_partial_progress()

    def evaluate_partial_progress(self) -> float:
        """
        Evaluate the progress of the player to assign a partial reward.
        Returns:
            float: Reward based on intermediate progress.
        """
        max_health: int = self.game_context.max_hp
        current_health: int = self.game_context.cur_hp
        if max_health > 0:
            health_reward: float = current_health / max_health
            health_penalty: float = (max_health - current_health) / max_health
            return health_reward - 0.5 * health_penalty
        return -1.0

    @property
    def outcome(self) -> slaythespire.GameOutcome:
        """Access the outcome of the current game context."""
        return self.game_context.outcome
