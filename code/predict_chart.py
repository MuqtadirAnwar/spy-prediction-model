import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the trained model
model = joblib.load('../model/spy_model.pkl')

# Load intraday data for today
intraday_data = pd.read_csv('../data/spy_intraday_data.csv')

# Generate synthetic timestamps for tomorrow
today = datetime.now().date()
tomorrow = today + timedelta(days=1)
intraday_data['date'] = pd.to_datetime(intraday_data['date'])
intraday_data = intraday_data.sort_values('date')
time_intervals = intraday_data['date'].dt.time.unique()
synthetic_timestamps = [datetime.combine(tomorrow, t) for t in time_intervals]

# Limit the intraday data to the most recent 961 intervals
intraday_data = intraday_data.tail(961)

# Create features for prediction
features = intraday_data[['1. open', '2. high', '3. low', '4. close', '5. volume']]

# Standardize the features
scaler = StandardScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)

# Predict the closing prices for each interval tomorrow
predicted_prices = model.predict(features_scaled)

# Debug prints
print(f"Number of synthetic timestamps: {len(synthetic_timestamps)}")
print(f"Number of predicted prices: {len(predicted_prices)}")

# Create a DataFrame for the predicted prices
predicted_data = pd.DataFrame({
    'date': synthetic_timestamps,
    'predicted_close': predicted_prices
})

# Plot the predicted intraday prices for tomorrow
plt.figure(figsize=(10, 6))
plt.plot(predicted_data['date'], predicted_data['predicted_close'], label='Predicted Close')
plt.xlabel('Time')
plt.ylabel('Predicted Closing Price')
plt.title('Predicted Intraday Prices for SPY Tomorrow')
plt.legend()
plt.savefig('../visualizations/predicted_intraday_tomorrow.png')
plt.show()

print("Predicted intraday prices for SPY tomorrow have been plotted and saved as 'predicted_intraday_tomorrow.png'.")
