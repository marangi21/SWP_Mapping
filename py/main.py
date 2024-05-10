import torch
from torch.utils.data import DataLoader
from ViTRegressor import ViTRegressor
import TreeDataset
from math import sqrt
from tqdm import tqdm
from utilities import train_model, plot_best_losses
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri con Optuna')
    parser.add_argument('--batch_size', type=int, default=30, help='Dimensione del batch')
    parser.add_argument('--n_epochs', type=int, default=200, help='Numero di epoche di addestramento')
    args = parser.parse_args()  

    #usa gpu se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") 
    
    #iperparametri
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    num_hidden_layers = 2
    size_hidden_layers = [519, 948]
    learning_rate = 1e-5
    dropout_rates = [0.37349024149018334, 0.5573776200668319]

    #creazione modello
    model = ViTRegressor(num_hidden_layers, size_hidden_layers, dropout_rates, device)
    
    # Congelamento pesi encoder
    frozen_layers = ["vit_layer"] 
    for name, param in model.named_parameters():
        if name.split(".")[0] in frozen_layers:
            param.requires_grad = False
    model.vit_layer.vit_layer.conv_proj.weight.requires_grad = True
    model.vit_layer.vit_layer.conv_proj.bias.requires_grad = True
    model = model.to(device)
    
    # Crea datasets e dataloaders
    train_set = TreeDataset.TreeDataset(mode='train', augment = True)
    val_set = TreeDataset.TreeDataset(mode='val')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_set = TreeDataset.TreeDataset(mode='test')
    test_loader = DataLoader(test_set, batch_size=batch_size)
 
    #training/validadion
    best_train_loss, best_val_loss, avg_train_losses, avg_val_losses, early_stop_epoch = train_model(model=model, 
                                                                                    num_epochs=n_epochs, 
                                                                                    lr=learning_rate, 
                                                                                    train_loader=train_loader, 
                                                                                    val_loader=val_loader,
                                                                                    device=device)
    
    #testing
    model.load_state_dict(torch.load(r'C:\Users\giova\OneDrive\Desktop\projects\SWP-regr\best_model.pt'))
    model.eval()
    test_losses = []
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.unsqueeze(1).float().to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            test_losses.append(loss.item())
    test_loss = np.mean(test_losses)

    loss_lists = {'train_losses': avg_train_losses, 
                  'val_losses' : avg_val_losses,
                  'test_loss': test_loss,
                  'early_stop_epoch': early_stop_epoch}                  

    train_mse = best_train_loss
    train_rmse = sqrt(train_mse)
    val_mse = best_val_loss
    val_rmse = sqrt(val_mse)
    test_mse = test_loss
    test_rmse = sqrt(test_mse)

    print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}")
    print(f"Val MSE: {val_mse}, Val RMSE: {val_rmse}")
    print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}")

    plot_best_losses(loss_lists['train_losses'], 
                     loss_lists['val_losses'], 
                     loss_lists['test_loss'], 
                     loss_lists['early_stop_epoch'])


