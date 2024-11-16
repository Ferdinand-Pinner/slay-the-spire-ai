import torch
import torch.nn as nn
import torch.optim as optim
import lib.slaythespire as slaythespire

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 64)          # Hidden layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc2(x))  # ReLU activation
        x = torch.softmax(self.fc3(x), dim=-1)  # Softmax for probabilities
        return x
    
    def predict_policy(self, state):
        # Assuming your model uses a neural network to predict action probabilities
        # Forward pass through the network to get action logits (raw scores)
        action_logits = self.model.forward(state)  # Assuming `forward` runs the state through the network
        return action_logits
    
    def getOberservation(self, game_context):
        # Assuming game_context has a method to get the current state
        # This method should return a tensor representing the game state
        return slaythespire.NNInterface.getObservation(game_context)
    
    

# Example model initialization
input_size = 412  # Should match your observation space size
output_size = 10  # Should match the number of possible actions
model = PolicyNetwork(input_size, output_size)

# Example optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)