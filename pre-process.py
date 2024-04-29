import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read data from CSV
data = pd.read_csv('^NSEI.csv')

# Cleaning
# Check for missing values
print("Missing values:\n", data.isnull().sum())
# Check for duplicates
print("\nDuplicates:\n", data.duplicated().sum())

# No missing values or duplicates found, so we proceed to normalization and feature engineering

# Normalization
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

# Feature Engineering
data['Price_Volatility'] = data['High'] - data['Low']
data['Short_MA'] = data['Close'].rolling(window=10).mean()
data['Long_MA'] = data['Close'].rolling(window=50).mean()
data['Price_Momentum'] = data['Close'].pct_change()
data['Day_of_Week'] = pd.to_datetime(data['Date']).dt.dayofweek
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Year'] = pd.to_datetime(data['Date']).dt.year

# Drop NaN values after feature engineering
data.dropna(inplace=True)

# Print or visualize the cleaned, normalized, and engineered features
print(data.head())
