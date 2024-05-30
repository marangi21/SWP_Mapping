import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.vision_transformer import vit_b_16
import torch.nn.functional as F
from ViT11 import ViT11

class ViTRegressor(nn.Module):

    def __init__(self, num_hidden_layers, size_hidden_layers, dropout_rates, device):
        super(ViTRegressor, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.dropout_rates = dropout_rates
        self.device = device

        #ViT che accetta immagini a 11 canali in input (mlp head rimossa)
        self.vit_layer = ViT11()

        #MLP head for regression
        # Input layer
        self.input_layer = nn.Linear(768, size_hidden_layers[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers - 1):  # Loop per ogni hidden layer
            self.hidden_layers.append(nn.Linear(size_hidden_layers[i], size_hidden_layers[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(size_hidden_layers[-1], 1)# regression task
        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_rates[i]) for i in range(len(dropout_rates))])

        

    def forward(self, x: Tensor):
        x = x.to(next(self.parameters()).device)
        x = self.vit_layer(x)
            #ToDo: qui (post ViT) aggiungere info di temp e umid (128 nodi da un altro mlp)
        x = F.relu(self.input_layer(x))
        x = self.dropouts[0](x)  # applica dropout con dropout_rates[0]
        for i in range(self.num_hidden_layers - 1):  # loop per ogni hidden layer
            x = F.relu(self.hidden_layers[i](x))
            x = self.dropouts[i+1](x)  # dropout per quell'hidden layer
        x = self.output_layer(x)
        return x