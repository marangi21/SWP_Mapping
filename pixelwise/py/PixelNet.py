import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class PixelNet(nn.Module):

    def __init__(self, in_count, num_hidden_layers, size_hidden_layers, dropout_rates, device):
        super(PixelNet, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.dropout_rates = dropout_rates
        self.device = device

        #MLP for regression
        # Input layer
        self.input_layer = nn.Linear(in_count, size_hidden_layers[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers - 1):  # Loop per ogni hidden layer
            self.hidden_layers.append(nn.Linear(size_hidden_layers[i], size_hidden_layers[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(size_hidden_layers[-1], 1)# regression task
        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_rates[i]) for i in range(len(dropout_rates))])

    def forward(self, x: Tensor):
        x = F.relu(self.input_layer(x))
        for i in range(self.num_hidden_layers - 1):  # loop per ogni hidden layer
            x = F.relu(self.hidden_layers[i](x))
            x = self.dropouts[i+1](x)  # dropout per quell'hidden layer
        x = self.output_layer(x)
        return x