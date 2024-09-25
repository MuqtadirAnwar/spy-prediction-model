import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file_path):
    data = pd.read_csv(file_path, index_col='date', parse_dates=True)
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    # Load and preprocess data
    data = load_data('../data/spy_historical_data.csv')
    scaled_data, scaler = preprocess_data(data)

    # Create dataset
    time_step = 100
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    # Build and train model
    model = build_model()
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

    # Evaluate model
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train = scaler.inverse_transform([Y_train])
    Y_test = scaler.inverse_transform([Y_test])

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data['close'], label='Actual Price')
    plt.plot(data.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
    plt.plot(data.index[len(train_predict) + (time_step * 2) + 1:len(data) - 1], test_predict, label='Test Predict')
    plt.legend()
    plt.show()
