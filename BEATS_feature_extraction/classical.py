import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from module.utils import evaluate
from sklearn.preprocessing import StandardScaler

# Load features and labels from the .npy file
features = np.load('/DATA/arora8/SpeechUnderstanding/MinorProject/beats/results/representations/representations.npy', allow_pickle=True)
labels = np.load('/DATA/arora8/SpeechUnderstanding/MinorProject/beats/results/representations/labels.npy', allow_pickle=True)

features = features.reshape(features.shape[0], -1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

normalize = StandardScaler()
normalize.fit(X_train)
X_train = normalize.transform(X_train)
X_test = normalize.transform(X_test)

# Train and evaluate RandomForestClassifier
print("Processing random forest classifier")
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = evaluate(y_test, rf_pred)
print("RandomForestClassifier Accuracy:", rf_accuracy)

# Train and evaluate SVC
print("processing svm")
svc_clf = SVC()
svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)
svc_accuracy = evaluate(y_test, svc_pred)
print("SVC Accuracy:", svc_accuracy)

# Train and evaluate KNeighborsClassifier
print("processing knn")
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
knn_accuracy = evaluate(y_test, knn_pred)
print("KNeighborsClassifier Accuracy:", knn_accuracy)

# Train and evaluate LogisticRegression
print("processing logistic regression")
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_accuracy = evaluate(y_test, lr_pred)
print("LogisticRegression Accuracy:", lr_accuracy)
