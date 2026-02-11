import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

def make_sequences(X: np.ndarray, y: np.ndarray, window=30):
    seq_X, seq_y = [], []
    for i in range(len(X) - window):
        seq_X.append(X[i:i+window])
        seq_y.append(y[i+window])
    return np.array(seq_X), np.array(seq_y)

def build_lstm(input_shape, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(data_path):
    df = pd.read_csv(data_path)
    df['return'] = df['Close'].pct_change().shift(-1)  # next-day return target
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'macro_value', 'rsi_14', 'sma_10', 'vol_10']
    df = df.dropna(subset=features)
    
    X = df[features].values
    y = df['return'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    window = 30
    seq_X, seq_y = make_sequences(X_scaled, y, window)

    split_idx = int(len(seq_X) * 0.8)
    X_train, X_val = seq_X[:split_idx], seq_X[split_idx:]
    y_train, y_val = seq_y[:split_idx], seq_y[split_idx:]

    model = build_lstm(input_shape=(window, X_train.shape[2]))

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[es])

    model.save('models/lstm_model.h5')
    print("Model trained and saved!")

if __name__ == '__main__':
    train_lstm_model('data/processed/AAPL_with_macro.csv')
