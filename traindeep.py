import torch
import librosa
import numpy as np
import pandas as pd

from module.models import DeepLearningModel
DATA_PATH = '../UrbanSound8K/'

class AudioData:
    def __init__(self, DATA_PATH, folds, cropLength = 16000):
        self.DATA_PATH = DATA_PATH
        self.df = pd.read_csv(DATA_PATH + 'UrbanSound8K.csv')
        self.cropLength = cropLength
        self.folds = folds
        self.files = None
        self.labels = None
        self._read_files()

    def _read_files(self):
        self.df['fullPath'] = self.DATA_PATH + 'fold' + self.df['fold'].astype(str) + '/' + self.df['slice_file_name'].astype(str)
        self.df = self.df[self.df['fold'].isin(self.folds)]
        self.files = self.df['fullPath'].values
        self.labels = self.df['classID'].values
        self.labels = torch.tensor(self.labels).long()
    
    def _load_data(self, file):
        waveform, sr = librosa.load(file, sr=None)
        if len(waveform) < self.cropLength:
            waveform = np.pad(waveform, (0, self.cropLength - len(waveform)))
        else:
            waveform = waveform[:self.cropLength]
        waveform = torch.tensor(waveform).float()
        return waveform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        waveform = self._load_data(self.files[idx])
        label = self.labels[idx]
        return waveform, label

    
model = DeepLearningModel('configs/config1.txt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = AudioData(DATA_PATH, [1,2,3,4,5,6,7,8])
valData = AudioData(DATA_PATH, [9,10])
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
valoader = torch.utils.data.DataLoader(valData, batch_size=32, shuffle=False)

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

epochs = 50
criterion = torch.nn.CrossEntropyLoss()

model.to(device)
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
valAcc = 0
bestvalAcc = 0
for epoch in range(epochs):
    model.train()
    for i, (waveforms, labels) in enumerate(dataloader):
        waveforms = waveforms.to(device)
  
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    print(f'Train Accuracy: {accuracy(model, dataloader, device)}')
    valAcc = accuracy(model, valoader, device)
    print(f'Val Accuracy: {valAcc}')

    if valAcc > bestvalAcc:
        bestvalAcc = valAcc
        torch.save(model.state_dict(), 'bestModel.pt')
        print('Model Saved')