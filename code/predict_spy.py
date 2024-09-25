import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load historical data
historical_data = pd.read_csv('../data/spy_historical_data.csv')

# Convert date column to datetime
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Sort data by date
historical_data = historical_data.sort_values('date')

# Create features and labels
historical_data['target'] = historical_data['4. close'].shift(-1)
features = historical_data[['1. open', '2. high', '3. low', '4. close', '5. volume']].iloc[:-1]
labels = historical_data['target'].iloc[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the model
import joblib
joblib.dump(model, '../model/spy_model.pkl')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()

print("Model training, evaluation, and visualization completed.")
