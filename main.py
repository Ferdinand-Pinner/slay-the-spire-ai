import random
import lib.slaythespire as slaythespire
from gamecontextwrapper import GameContextWrapper
from agent.agent import Agent  # MCTS with policy and value network

def main(ascension, character):
    # Initialize game context, neural network, and agent with MCTS & NN
    bleh = slaythespire.GameContext(character, random.randint(1, 1000000), ascension)
    game_context = GameContextWrapper(bleh)
    neural_network = slaythespire.getNNInterface()
    agent = Agent(neural_network)

    # Loop through episodes for AlphaZero-style training
    for episode in range(1):        
        game_context_wrapper =  GameContextWrapper(bleh) # Reset environment for each episode
        episode_memory = []
        
        # Run the game until outcome is decided
        while game_context.outcome == slaythespire.GameOutcome.UNDECIDED:
            state = neural_network.getObservation(game_context_wrapper.game_context)

            action = agent.mcts_search(game_context)

            game_context_wrapper.step(action)
            value, policy_logits = agent.evaluate(state)

            # Store in memory for training
            episode_memory.append((state, policy_logits, value))

        # Episode complete, get reward and train network
        reward = agent.calculate_reward(game_context)
        agent.update_network(episode_memory, reward)
        print(f"Episode {episode} complete. Outcome: {game_context.outcome}")
    
    print("Training complete.")


if __name__ == "__main__":
    ascension = 10
    character = slaythespire.CharacterClass.IRONCLAD
    main(ascension, character)
