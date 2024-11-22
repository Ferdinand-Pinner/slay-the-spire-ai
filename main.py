import random
from typing import List, Tuple
from agent.model import PolicyNetwork
import lib.slaythespire as slaythespire
from gameContextNode import GameContextNode
from agent.agent import Agent  # MCTS with policy and value network

def main(ascension: int, character: slaythespire.CharacterClass) -> None:
    """
    Main function to train the agent using AlphaZero-style reinforcement learning.

    Args:
        ascension (int): The ascension level for the game.
        character (slaythespire.CharacterClass): The character class to use in the game.
    """
    # Initialize game context, neural network, and agent with MCTS & NN
    game_context = slaythespire.GameContext(character, random.randint(1, 1000000), ascension)
    nn_interface = slaythespire.getNNInterface()
    agent = Agent(model=PolicyNetwork(), nn_interface=nn_interface)

    # Loop through episodes for AlphaZero-style training
    for episode in range(1):  # Replace with desired number of episodes
        game_context_node = GameContextNode(game_context)  # Reset environment for each episode
        episode_memory: List[Tuple[List[float], List[float], float]] = []

        # Run the game until outcome is decided
        while game_context_node.is_non_terminal():
            state: List[float] = nn_interface.getObservation(game_context_node.game_context)

            action = agent.mcts_search(game_context_node)
            game_context_node.perform_action(action)

            value, policy_logits = agent.evaluate(game_context_node)

            # Store state, policy, and value in memory for training
            episode_memory.append((state, policy_logits.tolist(), value))

        # Episode complete, calculate reward, and train the network
        reward = game_context_node.calculate_reward()
        agent.update_network(episode_memory, reward)
        print(f"Episode {episode} complete. Outcome: {game_context_node.outcome}")

    print("Training complete.")


if __name__ == "__main__":
    ascension = 10  # Ascension level
    character = slaythespire.CharacterClass.IRONCLAD  # Selected character class
    main(ascension, character)
