import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
    
def train_model(model, num_epochs, lr, train_loader, val_loader, device):
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    patience = 20
    best_val_loss = np.inf 
    early_stop_epoch = 0
    count = 0 #early stop counter

    #ciclo di addestramento
    with tqdm(total=num_epochs, desc="Training.. ", leave=False) as epoch_pbar:
        for epoch in range(num_epochs):
            epoch_pbar.set_description(f"Epoch {epoch}/{num_epochs}")
            #training loop
            model.train()
            for X, y in train_loader:
                X = X.to(device)
                y = y.unsqueeze(1).float().to(device)
                #Forward pass
                y_pred = model(X)
                loss = loss_function(y_pred, y)
                #Backward pass e ottimizzazione
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            #Validation loop
            model.eval()
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y = y.unsqueeze(1).float().to(device)
                    y_pred = model(X)
                    loss = loss_function(y_pred, y)
                    val_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_val_losses.append(valid_loss)

            #print(f'[{epoch+1}/{num_epochs}]' +
            #    f'Train loss: {train_loss:.5f}' + 
            #    f'Val loss: {valid_loss:.5f}')
            
            epoch_pbar.update(1)
            epoch_pbar.set_postfix(train_loss=train_loss, val_loss=valid_loss)

            #clear per la prossima epoca
            train_losses = []
            val_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_train_loss = train_loss
                early_stop_epoch = epoch+1
                # At this point also save a snapshot of the current model
                #print("loss decreased. saving model...")
                torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'best_model.pt')) 
                count = 0
            else:
                #increment patience counter
                count+=1
                print(f'EarlyStopping counter: {count}/{patience}')
                if count == patience:
                    print("Early Stopping training..")
                    break

    return best_train_loss, best_val_loss, avg_train_losses, avg_val_losses, early_stop_epoch

def plot_best_losses(best_train_losses, best_val_losses, test_loss, early_stop_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(best_train_losses, label='Training Loss')
    plt.plot(best_val_losses, label='Validation Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.axvline(early_stop_epoch, color='r', linestyle='--', label='Early Stopping')
    plt.title('Training, Validation & Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_regression(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Plot')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'regression_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()