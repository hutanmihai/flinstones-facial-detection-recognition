import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from src.constants import CNN_TRAIN_IMAGES_PATH, CNN_VALIDATION_IMAGES_PATH, MODEL_PATH
from src.utils.helpers import show_image

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
trainImages = ImageFolder(CNN_TRAIN_IMAGES_PATH, transform=transform)
valImages = ImageFolder(CNN_VALIDATION_IMAGES_PATH, transform=transform)
trainDataLoader = DataLoader(trainImages, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valImages, batch_size=BATCH_SIZE)
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input images are 40x40x3 (BGR)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 40x40x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 20x20x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 10x10x64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # 5x5x128
        self.pool = nn.MaxPool2d(2, 2)
        # 2x2x128
        self.fc1 = nn.Linear(2 * 2 * 128, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 2 * 2 * 128)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


model = CNN().to(device)
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.BCELoss()
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
print("[INFO] training the network...")
startTime = time.time()

for e in range(EPOCHS):
    model.train()
    totalTrainLoss = 0
    trainCorrect = 0

    for x, y in trainDataLoader:
        opt.zero_grad()
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        y_one_hot = torch.zeros((y.size(0), 2))  # Assuming 2 classes
        y_one_hot[torch.arange(y.size(0)), y] = 1  # One-hot encoding
        y_one_hot = y_one_hot.to(device)
        loss = lossFn(pred, y_one_hot)
        loss.backward()
        opt.step()
        totalTrainLoss += loss.item()

        pred_binary = (pred > 0.5).float()
        trainCorrect += (pred_binary.argmax(dim=1) == y).sum().item()

    avgTrainLoss = totalTrainLoss / trainSteps
    trainAccuracy = trainCorrect / len(trainDataLoader.dataset)

    model.eval()
    totalValLoss = 0
    valCorrect = 0

    with torch.no_grad():
        for x, y in valDataLoader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            y_one_hot = torch.zeros((y.size(0), 2))  # Assuming 2 classes
            y_one_hot[torch.arange(y.size(0)), y] = 1  # One-hot encoding
            y_one_hot = y_one_hot.to(device)
            loss = lossFn(pred, y_one_hot).item()
            pred_binary = (pred > 0.5).float()
            valCorrect += (pred_binary.argmax(dim=1) == y).sum().item()

    avgValLoss = totalValLoss / valSteps
    valAccuracy = valCorrect / len(valDataLoader.dataset)

    H["train_loss"].append(avgTrainLoss)
    H["train_acc"].append(trainAccuracy)
    H["val_loss"].append(avgValLoss)
    H["val_acc"].append(valAccuracy)

    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainAccuracy))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valAccuracy))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

torch.save(model, MODEL_PATH / "model.pth")
