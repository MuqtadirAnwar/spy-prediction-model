# SPY Stock Price Prediction

This project aims to predict the intraday and next-day closing prices for the SPDR S&P 500 ETF Trust (SPY) using various machine learning techniques.

## Features

- Fetches historical and intraday data for SPY using Alpha Vantage API
- Trains a linear regression model to predict stock prices
- Trains an LSTM (Long Short-Term Memory) model for more advanced time series prediction
- Predicts intraday prices for the next trading day
- Predicts the closing price for the next trading day
- Visualizes predictions and model performance

## Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- Alpha Vantage API
- TensorFlow (for LSTM model)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/MuqtadirAnwar/spy-prediction-model.git
   cd spy-prediction-model
   ```

2. Install the required dependencies:

   ```
   pip3 install pandas numpy scikit-learn matplotlib joblib alpha_vantage tensorflow
   ```

3. Set up your Alpha Vantage API key:
   - Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/)
   - Replace `'YOURAPIKEY'` in `fetch_spy_data.py` with your API key

## Usage

1. Fetch the latest SPY data:

   ```
   python3 fetch_spy_data.py
   ```

2. Train the linear regression prediction model:

   ```
   python3 predict_spy.py
   ```

3. Train the LSTM prediction model:

   ```
   python3 train_lstm.py
   ```

4. Predict tomorrow's closing price:

   ```
   python3 predict_tomorrow.py
   ```

5. Predict intraday prices for tomorrow:
   ```
   python3 predict_chart.py
   ```

## File Descriptions

- `fetch_spy_data.py`: Fetches historical and intraday data for SPY using Alpha Vantage API
- `predict_spy.py`: Trains the linear regression model and evaluates its performance
- `predict_tomorrow.py`: Predicts the closing price for the next trading day
- `predict_chart.py`: Predicts intraday prices for the next trading day and generates a chart
- `train_lstm.py`: Trains an LSTM model for time series prediction of SPY prices

## Output

- `spy_intraday_data.csv`: Intraday data for SPY
- `spy_historical_data.csv`: Historical daily data for SPY
- `spy_model.pkl`: Trained linear regression model
- `actual_vs_predicted.png`: Plot comparing actual vs predicted prices
- `feature_importances.png`: Bar chart showing the importance of each feature in the model
- `predicted_intraday_tomorrow.png`: Chart of predicted intraday prices for the next trading day
- LSTM model visualization (generated by `train_lstm.py`)

## Models

### Linear Regression

The project uses a simple linear regression model to predict stock prices based on historical data. This model is trained in `predict_spy.py`.

### LSTM (Long Short-Term Memory)

An LSTM model is implemented in `train_lstm.py` for more advanced time series prediction. LSTM is a type of recurrent neural network capable of learning long-term dependencies, making it well-suited for stock price prediction.

The LSTM model:

- Uses 100 time steps to predict the next price
- Has two LSTM layers with 50 units each, followed by a Dense layer
- Is trained on 80% of the available data
- Outputs visualizations comparing actual prices with predicted prices for both training and testing data

### Model scores

**Mean Squared Error:** 6.350823456758687

**R^2 Score:** 0.9995506180206406

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License.

## Disclaimer

This project is for educational purposes only. The predictions made by these models should not be used for actual trading decisions. Always do your own research and consult with a financial advisor before making investment decisions.
