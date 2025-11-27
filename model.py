import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

class Forecasting_Model:

    def __init__(self):
        self.data = pd.read_csv('stock_data.csv')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model placeholders
        self.model = None
        self.hybrid_model = None
        
        # Data placeholders (Loaded ONCE)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Scalers & Arrays
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_test_xgb = None
        self.y_test_xgb = None

    def create_windows(self, window_size=10):
        tickers = self.data['Ticker'].unique()
        combined_df = self.data.copy()
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            combined_df = combined_df.set_index('Date').sort_index()

        X_data_list = []
        y_data_list = []

        print("Creating windows")
        for ticker in tickers:
            ticker_df = combined_df[combined_df['Ticker'] == ticker].drop(columns=['Ticker'])
            
            # Ensure numeric
            try:
                X_ticker = ticker_df.drop(columns=['Target']).values.astype(float)
                y_ticker = ticker_df['Target'].values.astype(float)
            except:
                continue

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
    
    def prepare_data(self):
        X, y = self.create_windows(window_size=30)
        
        total_len = len(X)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        # Split raw data
        X_train_raw = X[:train_end]
        y_train_raw = y[:train_end]
        X_val_raw = X[train_end:val_end]
        y_val_raw = y[train_end:val_end]
        X_test_raw = X[val_end:]
        y_test_raw = y[val_end:]

        # Reshape for scaling
        n_features = X_train_raw.shape[2]
        
        # Fit Scalers ONLY on Training Data
        self.X_scaler.fit(X_train_raw.reshape(-1, n_features))
        self.y_scaler.fit(y_train_raw.reshape(-1, 1))

        # Transform
        def scale_X(data):
            return self.X_scaler.transform(data.reshape(-1, n_features)).reshape(data.shape)
        
        X_train = scale_X(X_train_raw)
        X_val = scale_X(X_val_raw)
        X_test = scale_X(X_test_raw)
        
        y_train = self.y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
        y_val = self.y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        y_test = self.y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

        # Create Loaders (Saved to Self)
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(self.device), 
                                           torch.from_numpy(y_train).float().to(self.device)), 
            batch_size=64, shuffle=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(self.device), 
                                           torch.from_numpy(y_val).float().to(self.device)), 
            batch_size=64, shuffle=False
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(self.device), 
                                           torch.from_numpy(y_test).float().to(self.device)), 
            batch_size=64, shuffle=False
        )
        
        print("✅ Data Preparation Complete.")

    def setup_model(self, input_size, hidden_size, num_layers, output_size):
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.dropout = nn.Dropout(0.2)
                self.fc1 = nn.Linear(hidden_size, hidden_size//2)
                self.tanh = nn.Tanh()
                self.fc2 = nn.Linear(hidden_size//2, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                last_step = out[:, -1, :]  # (batch_size, hidden_size)
                # Head
                out = self.fc1(last_step)
                out = self.tanh(out)
                out = self.dropout(out)
                prediction = self.fc2(out)
                return prediction, last_step  # Return context for XGBoost
            
        return LSTM(input_size, hidden_size, num_layers, output_size).to(self.device)
    
    def train_LSTM_model(self, num_epochs=30, learning_rate=0.001):
        print("\n--- Phase 1: Training LSTM ---")
        
        # Get input size from a sample batch
        sample_X, _ = next(iter(self.train_loader))
        input_size = sample_X.shape[2]

        self.model = self.setup_model(input_size, hidden_size=128, num_layers=3, output_size=1)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                preds, _ = self.model(X_batch)
                loss = criterion(preds, y_batch.unsqueeze(1)) # Ensure shape match
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

        self.evaluate_LSTM_model()

    def extract_features(self, loader):
            self.model.eval()
            features = []
            targets = []
            with torch.no_grad():
                for X_b, y_b in loader:
                    _, context = self.model(X_b)
                    features.append(context.cpu().numpy())
                    targets.append(y_b.cpu().numpy())
            return np.vstack(features), np.concatenate(targets)
    
    def train_XGB_model(self):
        print("\n--- Phase 2: Training XGBoost ---")
        self.model.eval()

        print("Extracting features...")
        X_train_xgb, y_train_xgb = self.extract_features(self.train_loader)
        X_val_xgb, y_val_xgb = self.extract_features(self.val_loader)
        self.X_test_xgb, self.y_test_xgb = self.extract_features(self.test_loader)

        # xgb_model = xgb.XGBRegressor(
        #     n_estimators=2000,
        #     learning_rate=0.005,
        #     max_depth=6,
        #     subsample=0.6,
        #     colsample_bytree=0.8,
        #     n_jobs=-1,
        #     early_stopping_rounds=50,
        #     random_state=42
        # )
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,            # Increased from 4 to 5 (Trends are more complex than noise)
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,          # L1 Regularization (Keep this to kill bad features)
            reg_lambda=0.5,         # Reduced L2 slightly (allow for larger target values)
            n_jobs=-1,
            early_stopping_rounds=50,
            random_state=42
        )
        xgb_model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_val_xgb, y_val_xgb)],
            verbose=False
        )
        print("XGBoost Training Complete.")
        return xgb_model

    def evaluate_LSTM_model(self):
        print("\n--- Evaluating LSTM Baseline ---")
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X_b, y_b in self.test_loader:
                p, _ = self.model(X_b)
                preds.append(p.cpu().numpy())
                targets.append(y_b.cpu().numpy())
        
        preds = np.vstack(preds)
        targets = np.concatenate(targets).reshape(-1, 1)
        
        # INVERSE TRANSFORM
        real_preds = self.y_scaler.inverse_transform(preds)
        real_targets = self.y_scaler.inverse_transform(targets)

        # Debug Check
        print(f"Sample Preds: {real_preds[:5].flatten()}")
        
        da = np.mean(np.sign(real_preds) == np.sign(real_targets)) * 100
        rmse = np.sqrt(np.mean((real_targets - real_preds)**2))
        print(f"LSTM RMSE: {rmse:.6f} | LSTM DA: {da:.2f}%")

    def evaluate_hybrid_model(self):
        print("\n--- Evaluating Hybrid Model ---")
        preds = self.hybrid_model.predict(self.X_test_xgb).reshape(-1, 1)
        
        # INVERSE TRANSFORM
        real_preds = self.y_scaler.inverse_transform(preds)
        real_targets = self.y_scaler.inverse_transform(self.y_test_xgb.reshape(-1, 1))

        print(f"Sample Preds: {real_preds[:5].flatten()}")

        da = np.mean(np.sign(real_preds) == np.sign(real_targets)) * 100
        rmse = np.sqrt(np.mean((real_targets - real_preds)**2))
        print(f"Hybrid RMSE: {rmse:.6f} | Hybrid DA: {da:.2f}%")

    def get_feature_importance(self):
        
        def get_rmse(X_input, y_target):
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X_input).float().to(self.device), 
                                               torch.from_numpy(y_target).float().to(self.device)), 
                batch_size=64, shuffle=False
            )
            X_xgb, _ = self.extract_features(loader)
            preds = self.hybrid_model.predict(X_xgb).reshape(-1, 1)
            real_preds = self.y_scaler.inverse_transform(preds)
            real_targets = self.y_scaler.inverse_transform(y_target.reshape(-1, 1))

            return np.sqrt(np.mean((real_targets-real_preds)**2))
        
        X_val, y_val = self.val_loader.dataset.tensors[0].cpu().numpy(), self.val_loader.dataset.tensors[1].cpu().numpy()
        baseline_rmse = get_rmse(X_val, y_val)
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change',
                         'RSI', 'MACD', 'EMA_ratio', 'CCI', 'OBV', 'BB', 'ATR', 'AD']
        result = []
        num_features = X_val.shape[2]
        for i in range(num_features):
            X_temp = X_val.copy()
            np.random.shuffle(X_temp[:,:,i])
            shuffled_rmse = get_rmse(X_temp, y_val)
            importance = shuffled_rmse - baseline_rmse
            feat_name = feature_names[i] if i < len(feature_names) else f"Feat {i}"
            result.append((feat_name, importance))
            print(f"   {feat_name}: +{importance:.6f} (RMSE: {shuffled_rmse:.6f})")

        result.sort(key=lambda x: x[1], reverse=True)
        print("\n--- Final Feature Importance (Top is Most Critical) ---")
        for name, score in result:
            print(f"{name:<15} | Impact: {score:.6f}")

    def execute_pipeline(self):
        self.prepare_data() 
        self.train_LSTM_model(num_epochs=20, learning_rate=0.001)
        self.hybrid_model = self.train_XGB_model()
        self.evaluate_hybrid_model()
        self.get_feature_importance()
        torch.save(self.model.state_dict(), 'lstm_model.pth')
        self.hybrid_model.save_model('hybrid_model.json')
        joblib.dump(self.X_scaler, 'X_scaler.gz')
        joblib.dump(self.y_scaler, 'y_scaler.gz')

    

if __name__ == "__main__":
    forecasting_model = Forecasting_Model()
    forecasting_model.execute_pipeline()