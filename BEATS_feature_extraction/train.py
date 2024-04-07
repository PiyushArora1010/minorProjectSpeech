#required imports
import torch

from BEATs import BEATs, BEATsConfig
from module.utils import UrbanSound, evaluate

from sklearn.model_selection import train_test_split as tts

# Load the pre-trained checkpoints
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.train()  # Set the model to train mode

# Define your loss function (e.g., cross-entropy)
loss_function = torch.nn.CrossEntropyLoss()

# Define your optimizer (e.g., Adam optimizer)
optimizer = torch.optim.Adam(BEATs_model.parameters(), lr=0.001)

# Load your dataset
Folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
trainData = UrbanSound(Folds)
trainData.data, testData, trainData.labels, testLabels = tts(trainData.data, trainData.labels, test_size=0.2, random_state=42)

# Define your DataLoader (if needed)
# dataloader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)

# Define your training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainData):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = BEATs_model(inputs)

        # Calculate the loss
        loss = loss_function(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
