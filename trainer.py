from module.utils import UrbanSound, ESC50Dataset, evaluate
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts

class trainer:
    def __init__(self, args):

        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

        if self.data == "urbansound":
            Folds = [1,2,3,4,5,6,7,8,9,10]
            self.trainData = UrbanSound(self.feature_type, Folds)
            self.trainData.data, self.testData, self.trainData.labels, self.testLabels = tts(self.trainData.data, self.trainData.labels, test_size=0.2, random_state=42)
        elif self.data == "esc50":
            self.trainData = ESC50Dataset(self.feature_type)
            self.trainData.data, self.testData, self.trainData.labels, self.testLabels = tts(self.trainData.data, self.trainData.labels, test_size=0.2, random_state=42)
            
    def _setModel(self):
        if self.model_type == "xgb":
            model = XGBClassifier()
        elif self.model_type == "svm":
            model = SVC()
        elif self.model_type == "rf":
            model = RandomForestClassifier()
        elif self.model_type == "knn":
            model = KNeighborsClassifier()
        elif self.model_type == "lr":
            model = LogisticRegression()
        else:
            raise ValueError("Invalid model type")
        return model

    def _train(self):
        model = self._setModel()
        self.mean, self.std = self.trainData.data.mean(axis=0), self.trainData.data.std(axis=0)
        self.trainData.data = (self.trainData.data - self.mean) / self.std
        self.testData = (self.testData - self.mean) / self.std
        model.fit(self.trainData.data, self.trainData.labels)
        return model
    
    def _evaluate(self, model):
        return evaluate(model, self.testData, self.testLabels)
    
    def run(self):
        model = self._train()
        results = self._evaluate(model)
        for key in results:
            print(f"{key}: {results[key]}")
