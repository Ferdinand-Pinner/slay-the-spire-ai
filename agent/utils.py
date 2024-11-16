import sys
sys.path.append('./lib')
import slaythespire

def process_game_state(game_context):
    """Converts game state to a fixed-size vector suitable for the neural network."""

    nn_interface = slaythespire.getNNInterface()

    # Assuming you have a GameContext object created elsewhere
    # Process the game state (e.g., convert to observation)
    observation = nn_interface.getObservation(game_context)


    return  observation
    
def log_rewards(reward):
    """Append rewards to a file for later analysis."""
    with open('./data/rewards_log.txt', 'a') as f:
        f.write(f"{reward}\n")