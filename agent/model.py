import torch
import torch.nn as nn
import torch.optim as optim
import lib.slaythespire as slaythespire

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Define the layers for your network
        self.fc1 = nn.Linear(412, 256)  # Adjust input size as needed
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 412)  # Action logits
        self.value_head = nn.Linear(64, 1)  # Scalar value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)  # Output scalar value
        return policy_logits, value

    def predict_policy(self, state):
        # Assuming state is a tensor
        _, policy_logits = self.forward(state)
        return policy_logits

    def predict_value(self, state):
        # Get the value output (scalar) from the value_head of the model
        _, value = self.forward(state)
        return value
    
    
    def getObservation(self, game_context):
        # Assuming game_context has a method to get the current state
        # This method should return a tensor representing the game state
        return slaythespire.NNInterface.getObservation(game_context)
