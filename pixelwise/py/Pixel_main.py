import torch
from torch.utils.data import DataLoader
from PixelNet import PixelNet
import PixelDataset
from math import sqrt
from tqdm import tqdm
from pixels_utilities import train_model, plot_best_losses, plot_regression
import argparse
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    num_hidden_layers = 9
    size_hidden_layers = [359, 556, 412, 976, 968, 926, 711, 566, 925]
    learning_rate = 0.0005
    dropout_rates = [0.66940799302729, 0.12602273793191196, 0.25424703566336665,
                     0.3004078921773323, 0.186443002734407, 0.47154497645078153,
                     0.46653028984884215, 0.324172442785381, 0.30115830818835654]

    # Crea datasets e dataloaders
    train_set = PixelDataset.PixelDataset(mode='train')
    val_set = PixelDataset.PixelDataset(mode='val')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_set = PixelDataset.PixelDataset(mode='test')
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    #creazione modello
    model = PixelNet(train_set.X.shape[1], num_hidden_layers, size_hidden_layers, dropout_rates, device)
    model.to(device)
 
    #training/validadion
    best_train_loss, best_val_loss, avg_train_losses, avg_val_losses, early_stop_epoch = train_model(model=model, 
                                                                                    num_epochs=n_epochs, 
                                                                                    lr=learning_rate, 
                                                                                    train_loader=train_loader, 
                                                                                    val_loader=val_loader,
                                                                                    device=device)
    
    #testing
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'best_model.pt')))
    model.eval()
    test_losses = []
    y_true = []
    y_hat = [] 
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.float().to(device)
            y = y.unsqueeze(1).float().to(device)
            y_pred = model(X)
            y_true.extend(y.cpu().numpy())
            y_hat.extend(y_pred.cpu().numpy())
            loss = loss_function(y_pred, y)
            test_losses.append(loss.item())
    test_loss = np.mean(test_losses)
    test_mse = mean_squared_error(y_true, y_hat)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_true, y_hat)

    loss_lists = {'train_losses': avg_train_losses, 
                  'val_losses' : avg_val_losses,
                  'test_loss': test_loss,
                  'early_stop_epoch': early_stop_epoch}                  

    train_mse = best_train_loss
    train_rmse = sqrt(train_mse)
    val_mse = best_val_loss
    val_rmse = sqrt(val_mse)

    print(f"Train MSE: {train_mse}, Train RMSE: {train_rmse}")
    print(f"Val MSE: {val_mse}, Val RMSE: {val_rmse}")
    print(f"Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R^2 score: {test_r2:.4f}")

    plot_best_losses(loss_lists['train_losses'], 
                     loss_lists['val_losses'], 
                     loss_lists['test_loss'], 
                     loss_lists['early_stop_epoch'])

    plot_regression(y_true, y_hat)
