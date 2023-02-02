import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

import dataGen

BATCH_SIZE = 128
LEARNING_RATE = 0.05
EPOCHS = 6000
DATA_FUNC = dataGen.cos


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


class Data(Dataset):
    """
    Data set
    """
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Generate data
data = DATA_FUNC()
xTrain = data[0][:50]
yTrain = data[1][:50]
xTest = data[0][50:]
yTest = data[1][50:]

# Create data set
trainSet = Data(xTrain, yTrain)
train_dataloader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)

testSet = Data(xTest, yTest)
test_dataloader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = Model()

def train(train_dl):
    # Loss function
    loss_fn = nn.MSELoss()
    # Optimizer to update weights
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    lossHistory = []

    for epoch in range(1, EPOCHS + 1):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
        
        for x, y in train_dl:
            # zero the parameter gradients
            optimizer.zero_grad()

            # propagate forward
            y_pred = model(x.unsqueeze(1))

            # calculate loss
            # unsqueeze to make y the same shape as y_pred
            loss = loss_fn(y_pred, y.unsqueeze(1))

            # propagate backward
            loss.backward()

            # update weights
            optimizer.step()
        
        lossHistory.append(loss.item())

    return lossHistory


def test(test_dl):
    # Calculate average accuracy
    loss = []
    for x, y in test_dl:
        y_pred = model(x.unsqueeze(1))
        loss.append(nn.MSELoss()(y_pred, y.unsqueeze(1)).item())
    
    return 1 - sum(loss) / len(loss)


lossHistory = train(train_dataloader)
accuracy = test(test_dataloader)
print(f"Test accuracy: {accuracy}")

# Plot data and loss
fig, (dataAx, lossAx) = plt.subplots(1, 2)
dataAx.scatter(xTrain, yTrain, label="Train data", color="green", s=10)
dataAx.scatter(xTest, yTest, label="Test data", color="blue", s=10)

xModel = np.linspace(-1, 1, 100)
yModel = model(torch.from_numpy(xModel).float().unsqueeze(1)).detach().numpy()

dataAx.plot(xModel, yModel, label="Model", color="red")
dataAx.legend()
dataAx.set_title("Data")

lossAx.plot(lossHistory, label="Loss", color="red")
lossAx.legend()
lossAx.set_title("Loss")

fig.tight_layout()
fig.suptitle(f"Model LR: {LEARNING_RATE} | Epochs: {EPOCHS} | Data: {DATA_FUNC.__name__} | Accuracy {accuracy*100:.2f}%")

plt.show()