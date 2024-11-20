import random
from agent.model import PolicyNetwork
import lib.slaythespire as slaythespire
from gameContextNode import GameContextNode
from agent.agent import Agent  # MCTS with policy and value network

def main(ascension, character):
    # Initialize game context, neural network, and agent with MCTS & NN
    bleh = slaythespire.GameContext(character, random.randint(1, 1000000), ascension)  
    NNinterface = slaythespire.getNNInterface()
    agent = Agent(PolicyNetwork(), NNinterface)

    # Loop through episodes for AlphaZero-style training
    for episode in range(1):        
        gameContextNode =  GameContextNode(bleh) # Reset environment for each episode
        episode_memory = []
        
        # Run the game until outcome is decided
        while gameContextNode.is_non_terminal():
            state = NNinterface.getObservation(gameContextNode.game_context)

            action = agent.mcts_search(gameContextNode)

            gameContextNode.step(action)
            value, policy_logits = agent.evaluate(state)

            # Store in memory for training
            episode_memory.append((state, policy_logits, value))

        # Episode complete, get reward and train network
        reward = agent.calculate_reward(gameContextNode)
        agent.update_network(episode_memory, reward)
        print(f"Episode {episode} complete. Outcome: {gameContextNode.outcome}")
    
    print("Training complete.")


if __name__ == "__main__":
    ascension = 10
    character = slaythespire.CharacterClass.IRONCLAD
    main(ascension, character)
