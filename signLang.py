import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# Hyperparameters
hidden_size = 200
num_classes = 26
num_epochs = 3
batch_size = 32
learning_rate = 2e-4
filter_size = 2

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        label = int(self.data_df.iloc[idx, 0])
        pixels = self.data_df.iloc[idx, 1:].values.astype('float32').reshape((28, 28))
        if self.transform:
            pixels = self.transform(pixels)
        return pixels, label

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Example normalization
])

# Initialize the dataset
train_dataset = SignLanguageDataset(csv_file='archive/sign_mnist_train/sign_mnist_train.csv', transform=data_transforms)
test_dataset = SignLanguageDataset(csv_file='archive/sign_mnist_test/sign_mnist_test.csv', transform=data_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class SignNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, fc1_size, fc2_size, fc3_size, num_classes):
        super(SignNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=filter_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=filter_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Example assuming input image size is (1, 28, 28):
        self.fc1 = nn.Linear(hidden_size*2*7*7, fc1_size)  # Adjusted size based on feature map dimensions
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc2_size)
        self.fc4 = nn.Linear(fc2_size, fc2_size)
        self.fc5 = nn.Linear(fc2_size, fc3_size)
        self.fc6 = nn.Linear(fc3_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

model = SignNeuralNetwork(hidden_size=hidden_size, filter_size=filter_size, fc1_size=100, fc2_size=100,fc3_size=80, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model(model):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

        # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()       # Backpropagation
            optimizer.step()      # Update parameters

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()  # Sum up the batch loss
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct += (predicted == labels).sum().item()  # Count the number of correct predictions

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


train_model(model)

test_model(model, test_loader, criterion, device)

torch.save(model, 'model_full.pth')
