import torch
from torch.utils.data import Dataset
import pandas as pd
import rasterio as rio
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import os

class TreeDataset(Dataset):

  def __init__(self, mode, augment=False):
    #data loading in base alla modalità
    if mode == 'train':
      data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train_set.csv'), header=0)
    elif mode == 'val':
      data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'val_set.csv'), header=0)    
    elif mode == 'test':
      data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test_set.csv'), header=0)   
    else:
      raise ValueError('Invalid mode')

    self.img_paths = data['img_path'] # lista di path delle immagini
    self.labels = data[' SWP'] # lista di float
    self.n_samples = len(self.labels)
    self.augment = augment

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    img_path = self.img_paths[index]
    with rio.open(img_path) as img:
      img_data = img.read()
    img_tensor = torch.from_numpy(img_data)

    # medie e stds per canale del train_set (calcolate precedentemente sul file binario)
    train_means = [17715.6875, 16900.486328125, 19324.916015625, 18412.6015625, 33659.69140625, 16722.482421875, 20676.611328125, 30762.6875, 26453.736328125, 16695.986328125, 43.55046844482422]
    train_stds = [10419.953125, 10096.2568359375, 10045.6787109375, 9415.35546875, 8845.2705078125, 11132.8017578125, 10644.0224609375, 9388.4013671875, 9778.1474609375, 11603.029296875, 7.6262030601501465]

    # Data augmentation opzionale
    if self.augment:
      train_transforms = v2.Compose([
                        v2.Resize((224, 224), antialias=True),
                        transforms.RandomApply([v2.RandomCrop(180)], p=0.5),
                        v2.Resize((224, 224), antialias=True),
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        #transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(train_means, train_stds)])
    else:
      train_transforms = v2.Compose([
                        v2.Resize((224, 224), antialias=True),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(train_means, train_stds)])
      
    img_tensor = train_transforms(img_tensor)
    return img_tensor, self.labels[index]