import torch
from torch.utils.data import DataLoader
from BEATs import BEATs, BEATsConfig

from module.utils import UrbanSound, ESC50Dataset, evaluate
from sklearn.model_selection import train_test_split as tts

import numpy as np

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#global variables
BATCH_SIZE = 32

# load the pre-trained checkpoints
checkpoint = torch.load('/DATA/arora8/SpeechUnderstanding/MinorProject/beats/checkpoints/BEATs_iter1.pt')
print("[checkpoint loaded]")

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()
print("[model set to eval]")

# extract the the audio representation
# audio_input_16khz = torch.randn(1, 10000)
# padding_mask = torch.zeros(1, 10000).bool()
Folds = [1,2,3,4,5,6,7,8,9,10]
trainset = UrbanSound(Folds)
# trainset = ESC50Dataset()
traindata = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
print("[dataloader is ready]")

#store extracted features
representation_list = []
labels_list = []

print("[extraction started]")
#extract features
for idx, data in enumerate(traindata):

    batch_data, labels = data
    padding_size = batch_data.size(0)

    print("Processing batch:", idx)

    padding_mask_batch = torch.zeros(padding_size, 16000).bool()
    padding_mask_batch = padding_mask_batch.to(device)

    batch_data = batch_data.to(device)

    representation = BEATs_model.extract_features(batch_data, padding_mask=padding_mask_batch)[0]
    representation_list.extend(np.array(representation.detach()))
    labels_list.extend(np.array(labels))

np.save("/DATA/arora8/SpeechUnderstanding/MinorProject/beats/results/representations/representations.npy", np.array(representation_list))
np.save("/DATA/arora8/SpeechUnderstanding/MinorProject/beats/results/representations/labels.npy", np.array(labels_list))
