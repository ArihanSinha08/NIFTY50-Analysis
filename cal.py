import pandas as pd
import numpy as np

# Read data from CSV
data = pd.read_csv('^NSEI.csv')

# Calculate Moving Averages (MA)
def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

data['MA_10'] = calculate_moving_average(data, 10)
data['MA_20'] = calculate_moving_average(data, 20)

# Calculate MACD
def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

data['MACD_line'], data['Signal_line'], data['MACD_histogram'] = calculate_macd(data, 12, 26, 9)

# Calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI_14'] = calculate_rsi(data, 14)

# Print or visualize the calculated indicators
print(data[['Date', 'Close', 'MA_10', 'MA_20', 'MACD_line', 'Signal_line', 'MACD_histogram', 'RSI_14']])
