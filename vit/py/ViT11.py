import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class ViT11(nn.Module):
    def __init__(self):
        super(ViT11, self).__init__()

        #caricamento modello pretrainato
        self.vit_layer = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')

        #calcolo pesi 11 canali
        rgb_weights = self.vit_layer.conv_proj.weight
        red = rgb_weights[:,0:1,:,:]
        green = rgb_weights[:,1:2,:,:]
        blue = rgb_weights[:,2:3,:,:]
        weights_11 = torch.cat([blue,
                                blue,
                                green,
                                green,
                                green,
                                red,
                                green,
                                green,
                                green,
                                red,
                                green], dim=1)

        #modifica primo layer per accettare immagini a 12 canali come input
        self.vit_layer.conv_proj = nn.Conv2d(11, 768, kernel_size=(16,16), stride=(16,16))

        #aggiornamento nuovi pesi a 12 canali
        self.vit_layer.conv_proj.weight = torch.nn.Parameter(weights_11)

        #rimozione mlp_head
        self.vit_layer.heads = nn.Identity()

    def forward(self, x):
        x = self.vit_layer(x)
        return x