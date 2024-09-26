from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def fetch_intraday_data(api_key, symbol='SPY', interval='1min', outputsize='full'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    return data

def fetch_historical_data(api_key, symbol='SPY', outputsize='full'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)
    return data

if __name__ == '__main__':
    api_key = 'YOURAPIKEY'
    intraday_data = fetch_intraday_data(api_key)
    historical_data = fetch_historical_data(api_key)

    # Save data to CSV files
    intraday_data.to_csv('../data/spy_intraday_data.csv') # Change File Path As Needed
    historical_data.to_csv('../data/spy_historical_data.csv') # Change File Path As Needed

    print("Intraday and historical data for SPY have been fetched and saved to CSV files.")
