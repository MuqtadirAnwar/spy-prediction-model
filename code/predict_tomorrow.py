import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('../model/spy_model.pkl')

# Load historical data
historical_data = pd.read_csv('../data/spy_historical_data.csv')

# Convert date column to datetime
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Sort data by date
historical_data = historical_data.sort_values('date')

# Get the most recent data point
latest_data = historical_data[['1. open', '2. high', '3. low', '4. close', '5. volume']].iloc[-1:]

# Standardize the features
scaler = StandardScaler()
scaler.fit(historical_data[['1. open', '2. high', '3. low', '4. close', '5. volume']])
latest_data_scaled = scaler.transform(latest_data)

# Make a prediction
predicted_price = model.predict(latest_data_scaled)

print(f"Predicted closing price for tomorrow: {predicted_price[0]}")

# Visualize the feature importances (coefficients)
coefficients = model.coef_
features = ['1. open', '2. high', '3. low', '4. close', '5. volume']

plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importances (Coefficients) of the Linear Regression Model')
plt.savefig('../visualizations/feature_importances.png')
plt.show()
