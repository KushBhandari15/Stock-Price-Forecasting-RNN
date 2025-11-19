from data_processing import get_all_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class Directonal_MSELoss(nn.Module):
    def __init__(self, alpha=10.0):
        super(Directonal_MSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        directional_error = torch.tanh(predictions) * torch.tanh(targets)
        direction_penalty = torch.mean(torch.relu(-1 * directional_error))
        return mse_loss + torch.mean(direction_penalty * self.alpha)
    
class Forecasting_Model:

    def __init__(self):
        self.data = pd.read_csv('stock_data.csv')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model = None

    def create_windows(self, window_size=10):
        
        tickers = self.data['Ticker'].unique()
        combined_df = self.data
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.set_index('Date').sort_index()

        X_data_list = []
        y_data_list = []

        for ticker in tickers:
            ticker_df = combined_df[combined_df['Ticker'] == ticker].drop(columns=['Ticker'])
            ticker_df.dropna(inplace=True)

            X_ticker = ticker_df.drop(columns=['Target']).values
            y_ticker = ticker_df['Target'].values

            if len(X_ticker) < window_size:
                continue

            N = len(X_ticker) - window_size 
            starts = np.arange(N)

            X_windows = np.array([X_ticker[i:i+window_size] for i in starts])
            Y_targets = y_ticker[window_size : window_size + N] 
            
            X_data_list.append(X_windows)
            y_data_list.append(Y_targets)

        X_final = np.concatenate(X_data_list, axis=0)
        y_final = np.concatenate(y_data_list, axis=0)

        return X_final, y_final
    
    def pre_processing(self):

        X, y = self.create_windows(window_size=10)
        total_len = len(X)
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        train_end_idx = int(total_len * train_ratio)
        val_end_idx = int(total_len * (train_ratio + val_ratio))

        X_train_raw = X[:train_end_idx]
        y_train_raw = y[:train_end_idx]
        X_val_raw = X[train_end_idx:val_end_idx]
        y_val_raw = y[train_end_idx:val_end_idx]
        X_test_raw = X[val_end_idx:]
        y_test_raw = y[val_end_idx:]

        n_features = X_train_raw.shape[2]
        X_train_2D = X_train_raw.reshape(-1, n_features)

        # Initialize scalers
        X_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))

        # Fit scalers
        X_scaler.fit(X_train_2D)
        y_scaler.fit(y_train_raw.reshape(-1, 1)) # y must be 2D for the scaler

        # Transform all sets
        X_train = X_scaler.transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
        X_val = X_scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
        X_test = X_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

        # y does not need reshaping after transformation, only before and during fit/transform
        y_train = y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
        y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        # 4. Convert to PyTorch Tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

        return (X_train_tensor, y_train_tensor,
                X_val_tensor, y_val_tensor,
                X_test_tensor, y_test_tensor,
                y_scaler)

    def setup_model(self, input_size, hidden_size, num_layers, output_size):

        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.attention_weights = nn.Linear(hidden_size, 1)
                self.fc1 = nn.Linear(hidden_size, hidden_size//2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(hidden_size//2, output_size)


            def forward(self, x):
                out, _ = self.lstm(x)
                attn_score = self.attention_weights(out)
                attn_weights = torch.softmax(attn_score, dim=1)
                context = torch.sum(attn_weights * out, dim=1)
                # out = out[:, -1, :]
                out = self.fc1(context)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return out
            
        return LSTM(input_size, hidden_size, num_layers, output_size).to(self.device)
    
    def train_model(self, num_epochs=50, learning_rate=0.001):

        X_train, y_train, X_val, y_val, _, _, _ = self.pre_processing()
        X_train = X_train.to(self.device); y_train = y_train.to(self.device)
        X_val = X_val.to(self.device); y_val = y_val.to(self.device)

        input_size = X_train.shape[2]
        self.model = self.setup_model(input_size, hidden_size=128, num_layers=2, output_size=1)
        criterion = nn.MSELoss()
        # criterion = Directonal_MSELoss(alpha=0.5)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        for epoch in range(num_epochs):
            self.model.train()
            total_training_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item() * X_batch.size(0)

            avg_training_loss = total_training_loss / len(train_dataset)

            total_val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    val_predictions = self.model(X_batch_val)
                    val_loss = criterion(val_predictions, y_batch_val)
                    total_val_loss += val_loss.item() * X_batch_val.size(0)
            
            avg_val_loss = total_val_loss / len(val_dataset)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_training_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        print(f"\nFinal Training Complete. Average Validation Loss: {avg_val_loss:.6f}")

    def evaluate_model(self):

        _, _, _, _, X_test, y_test, y_scaler = self.pre_processing()

        X_test = X_test.to(self.device); y_test = y_test.to(self.device)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

        self.model.eval()
        total_test_loss = 0.0
        test_predictions_list = []
        test_targets_list = []
        with torch.no_grad():
            for X_batch_test, y_batch_test in test_loader:
                test_predictions = self.model(X_batch_test)
                test_predictions_list.append(test_predictions.cpu().numpy())
                test_targets_list.append(y_batch_test.cpu().numpy())
                loss = nn.MSELoss()(test_predictions, y_batch_test)
                total_test_loss += loss.item() * X_batch_test.size(0)

        predictions_scaled = np.vstack(test_predictions_list)
        targets_scaled = np.vstack(test_targets_list)

        avg_test_loss = total_test_loss / len(test_dataset)
        print(f"\nFinal Test Loss (MSE on scaled data): {avg_test_loss:.6f}")

        predictions_real = y_scaler.inverse_transform(predictions_scaled)
        targets_real = y_scaler.inverse_transform(targets_scaled)
        rmse = np.sqrt(np.mean((targets_real - predictions_real)**2))
        print(f"Test RMSE (on scaled data): {rmse:.6f}")
        correct_direction = np.sign(predictions_real) == np.sign(targets_real)
        directional_accuracy = np.sum((correct_direction) / len(y_test))*100
        print(f"Test Directional Accuracy (DA): {directional_accuracy:.2f}%")

        return rmse, directional_accuracy

    def execute_pipeline(self):

        self.train_model(num_epochs=30, learning_rate=0.001)
        self.evaluate_model()
        torch.save(self.model.state_dict(), 'forecasting_model.pth')

if __name__ == "__main__":
    forecasting_model = Forecasting_Model()
    forecasting_model.execute_pipeline()