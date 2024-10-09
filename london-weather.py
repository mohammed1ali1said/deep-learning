import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data_filled = data.ffill()
    features = data_filled.drop(['date', 'min_temp', 'mean_temp', 'max_temp'], axis=1)
    targets = data_filled[['min_temp', 'mean_temp', 'max_temp']]

    # Normalize the features and targets
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaler = MinMaxScaler()
    targets_scaled = target_scaler.fit_transform(targets)

    return features_scaled, targets_scaled, target_scaler


def create_sequences(input_data, target_data, time_steps=7):
    X, y = [], []
    for i in range(0, len(input_data) - time_steps):
        input_seq = input_data[i:(i + time_steps), :]  # Extract input sequence
        target_seq = target_data[i + time_steps, :]  # Extract target sequence
        X.append(input_seq)
        y.append(target_seq)
    return np.array(X), np.array(y)

# Define LSTM model
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(hidden_layer_size, 70)
        self.fc2 = nn.Linear(70, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, output_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        predictions = self.fc4(x)

        return predictions


def train_model(model, train_loader, num_epochs, optimizer, loss_function):
    all_train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        all_train_losses.append(avg_train_loss)

        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')

    return all_train_losses


def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            test_predictions.extend(y_pred.cpu().numpy())
            test_actuals.extend(y_batch.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    # Inverse transform to get actual values
    test_predictions_inverse = target_scaler.inverse_transform(test_predictions)
    test_actuals_inverse = target_scaler.inverse_transform(test_actuals)

    # Calculate MSE for each target
    test_mse = {
        'min_temp': mean_squared_error(test_actuals_inverse[:, 0], test_predictions_inverse[:, 0]),
        'mean_temp': mean_squared_error(test_actuals_inverse[:, 1], test_predictions_inverse[:, 1]),
        'max_temp': mean_squared_error(test_actuals_inverse[:, 2], test_predictions_inverse[:, 2]),
    }

    return test_predictions_inverse, test_actuals_inverse, test_mse


def plot_results(actuals, predictions, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



# Main workflow
if __name__ == "__main__":
    
    file_path = 'london_weather.csv'
    features_scaled, targets_scaled, target_scaler = load_and_preprocess_data(file_path)

    # Create sequences
    X, y = create_sequences(features_scaled, targets_scaled)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    input_size = X.shape[2]
    hidden_layer_size = 100
    output_size = targets_scaled.shape[1]
    num_epochs = 60
    batch_size = 32
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = WeatherLSTM(input_size, hidden_layer_size, output_size).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_train_losses = train_model(model, train_loader, num_epochs, optimizer, loss_function)

    # Evaluate the model
    test_predictions_inverse, test_actuals_inverse, test_mse = evaluate_model(model, test_loader, target_scaler)

    # Print MSE results
    for key, value in test_mse.items():
        print(f'Test MSE for {key.replace("_", " ").title()}: {value}')

    # Plot results
    plot_results(test_actuals_inverse[:, 0], test_predictions_inverse[:, 0], 'Actual vs. Predicted Minimum Temperature', 'Minimum Temperature')
    plot_results(test_actuals_inverse[:, 1], test_predictions_inverse[:, 1], 'Actual vs. Predicted Mean Temperature', 'Mean Temperature')
    plot_results(test_actuals_inverse[:, 2], test_predictions_inverse[:, 2], 'Actual vs. Predicted Maximum Temperature', 'Maximum Temperature')


