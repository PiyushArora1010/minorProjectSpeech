import librosa
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torchaudio import load

class ClassicalFeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=20, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cropLengthAudios = 16000

    def __call__(self, waveform, feature_type):
        if feature_type == "mfcc":
            return self.extract_mfcc(waveform)
        elif feature_type == "log_mel":
            return self.extract_log_mel(waveform)
        elif feature_type == "stft":
            return self.extract_stft(waveform)
        else:
            print("Invalid feature type")
            exit(1)
        
    def extract_mfcc(self, waveform):
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfcc
    
    def extract_log_mel(self, waveform):
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel)
        return log_mel
    
    def extract_stft(self, waveform):
        stft = librosa.stft(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length)
        return np.abs(stft)

class UrbanSound:
    def __init__(self, feat_type, folds, cropLength = 16000):
        self.DATA_PATH = '../UrbanSound8K/'
        self.df = pd.read_csv(self.DATA_PATH + 'UrbanSound8K.csv')
        self.cropLength = cropLength
        self.folds = folds
        self.feature_type = feat_type
        self.files = None
        self.labels = None
        self._read_files()
        self.featExtractor = ClassicalFeatureExtractor()
        self.data = self._load_data(self.files)

    def _read_files(self):
        self.df['fullPath'] = self.DATA_PATH + 'fold' + self.df['fold'].astype(str) + '/' + self.df['slice_file_name'].astype(str)
        self.df = self.df[self.df['fold'].isin(self.folds)]
        self.files = self.df['fullPath'].values
        self.labels = self.df['classID'].values
    
    def _load_data(self, files):
        data = []
        for file in files:
            waveform, _ = librosa.load(file, sr=16000)
            if len(waveform) < self.cropLength:
                waveform = np.pad(waveform, (0, self.cropLength - len(waveform)), "constant")
            else:
                waveform = waveform[:self.cropLength]
            feature = self.featExtractor(waveform, self.feature_type)
            feature = np.mean(feature, axis=1)
            data.append(feature)
        data = np.array(data)
        return data

def evaluate(model, data, labels):
    dicresults = {}
    predictions = model.predict(data)
    dicresults['accuracy'] = accuracy_score(labels, predictions)
    dicresults['f1'] = f1_score(labels, predictions, average='weighted', zero_division=0)
    dicresults['precision'] = precision_score(labels, predictions, average='weighted', zero_division=0)
    dicresults['recall'] = recall_score(labels, predictions, average='weighted', zero_division=0)
    return dicresults

class ESC50Dataset:
    def __init__(self, feat_type):
        self.data_dir = '../ESC-50-master/'
        self.feature_type = feat_type
        self.metadata = pd.read_csv(self.data_dir + 'meta/' + 'esc50.csv')
        self.featExtractor = ClassicalFeatureExtractor()
        self.files = self.metadata['filename'].values
        self.labels = self.metadata['category'].values
        self.labels = pd.Categorical(self.labels).codes

        assert len(self.files) == len(self.labels)
        self.data = self._load_data(self.files)

    def _load_data(self, files):
        data = []
        for file in files:
            waveform, _ = librosa.load(self.data_dir + 'audio/' + file)
            feature = self.featExtractor(waveform, self.feature_type)
            feature = np.mean(feature, axis=1)
            data.append(feature)
        data = np.array(data)
        return data
