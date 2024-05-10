import torch
from torch.utils.data import DataLoader
from ViTRegressor import ViTRegressor
import TreeDataset
from math import sqrt
from tqdm import tqdm
import optuna
from optuna.study import StudyDirection
from optuna.samplers import TPESampler
from utilities import train_model, plot_best_losses
import os
import joblib
import argparse

loss_lists= {}

def objective(trial, batch_size, n_epochs, device, seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trial.set_user_attr("n_epochs", n_epochs)
    trial.set_user_attr("batch_size", batch_size)

    try:
        #suggerimento iperparametri
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
        size_hidden_layers = [trial.suggest_int(f'neurons_per_layer_{i}', 16, 1024) for i in range(num_hidden_layers)] 
        learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6])
        dropout_rates = [trial.suggest_float(f'dropout_rates_{i}', 0.1, 0.8) for i in range(num_hidden_layers)]
        num_epochs = n_epochs
        batch_size = batch_size

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

        #training
        best_val_loss, avg_train_losses, avg_val_losses, early_stop_epoch = train_model(model=model, 
                                                                                        num_epochs=num_epochs, 
                                                                                        lr=learning_rate, 
                                                                                        train_loader=train_loader, 
                                                                                        val_loader=val_loader,
                                                                                        device=device)
        

        #lista per plottare train/val loss della trial migliore
        loss_lists[trial.number] = (avg_train_losses, avg_val_losses, early_stop_epoch)

        return best_val_loss
    except Exception as e:
        print(f"An exception occurred during optimization: {e}")
        print("Saving study and exiting...")
        joblib.dump(study, r'C:\Users\CR-MASTER\Desktop\projects\SWP-regr\optuna_results.pkl')
        raise


if __name__ == "__main__":

    seed = 42
    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri con Optuna')
    parser.add_argument('--batch_size', type=int, default=30, help='Dimensione del batch')
    parser.add_argument('--n_trials', type=int, default=10, help='Numero di tentativi di ottimizzazione')
    parser.add_argument('--n_epochs', type=int, default=200, help='Numero di epoche di addestramento')
    args = parser.parse_args()  

    n_trials = args.n_trials
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    #usa una GPU se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") 

    sampler = TPESampler(seed=42)
    
    #Se lo studio non esiste, lo crea. Se esiste, lo carica e stampa la trial migliore finora.
    if not os.path.exists(r'C:\Users\CR-MASTER\Desktop\projects\SWP-regr\optuna_results.pkl'):
        study = optuna.create_study(direction=StudyDirection.MINIMIZE, sampler=sampler)
    else:
        study = joblib.load(r'C:\Users\CR-MASTER\Desktop\projects\SWP-regr\optuna_results.pkl')
        print("Best trial until now:")
        print("  Number: ", study.best_trial.number)
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.FAIL: 
                study.enqueue_trial(trial.params)
                print(f"enqueued previosly failed trial...{trial}")
    
    with tqdm(total=n_trials, desc="Optimizing.. ") as trial_pbar:
        def callback(study, trial):
            trial_pbar.set_description(f"Trial {trial.number+1}/{n_trials}")
            trial_pbar.update(1)
            joblib.dump(study, r'C:\Users\CR-MASTER\Desktop\projects\SWP-regr\optuna_results.pkl')
        
        study.optimize(lambda trial: objective(trial, batch_size, n_epochs, device, seed),
                        n_trials=n_trials,
                        callbacks=[callback])

    print(study.trials_dataframe())
    print(f'Number of finished trials: {len(study.trials)}')
    #print(f'Best trial: {study.best_trial}')
    print(f'Best MSE: {study.best_trial.value}')
    print(f'Best RMSE: {sqrt(study.best_trial.value)}')
    print(f'Best value: {study.best_value}')
    print(f'Best hyperparameters: {study.best_params}')

    #plot_best_losses(loss_lists[study.best_trial.number])


