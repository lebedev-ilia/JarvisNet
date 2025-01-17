import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm
import torch
from hparams import num_classes

class MyDataset(Dataset):
  def __init__(self, path):
    self.data = self.load_data(path)
    
  def load_data(self, path):
    data = []
    data_csv = pd.read_csv(path + 'metadata.csv')
    tqdm_r = tqdm(range(1000), desc='Loading data...')
    k = 0
    for i in tqdm_r:
      if k <= 32:
        subpath = data_csv['path'][k]
        classs = data_csv['class'][k]
        waveform, sample_rate = torchaudio.load(path + subpath)
        label = torch.zeros(num_classes)
        label[classs-1] = 1
        data.append(list((waveform, sample_rate, label)))
        k+=1
      else: k = 0
    return data
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, item):
    if self.data[item][0][-1].shape[0] < 63488:
      dop = torch.zeros(1, 63488 - self.data[item][0][-1].shape[0])
      self.data[item][0] = torch.cat((self.data[item][0], dop), dim=1)
    return self.data[item]