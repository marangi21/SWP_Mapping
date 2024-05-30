from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import joblib
import pandas as pd
import os

class PixelDataset(Dataset):

    def __init__(self, mode):
        #data loading in base alla modalità
        if mode == 'train':
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'px_train_set.csv'), header=0)
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset','scaler_pixelwise.pkl')):
                scaler = StandardScaler()
                scaler.fit(data.drop('SWP', axis=1))
                joblib.dump(scaler, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'scaler_pixelwise.pkl'))
        elif mode == 'val':
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'px_val_set.csv'), header=0)    
        elif mode == 'test':
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'px_test_set.csv'), header=0)   
        else:
            raise ValueError('Invalid mode')

        self.X = data.drop('SWP', axis=1).reset_index(drop=True) # features
        self.y = data['SWP'] # lista di float
        self.scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'scaler_pixelwise.pkl'))
        self.X = self.scaler.transform(self.X) # scaling features con scaler addestrato sul training set
        self.n_samples = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]