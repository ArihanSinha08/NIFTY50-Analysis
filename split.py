import pandas as pd
from sklearn.model_selection import train_test_split

# Read data from CSV
data = pd.read_csv('^NSEI.csv')

# Split the dataset into features (X) and target variable (y)
X = data.drop(['Close', 'Date'], axis=1)  # Drop 'Close' column and 'Date' column (if not used as a feature)
y = data['Close']

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save datasets to CSV files
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=['Close'])  # Save target variable with a header
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False, header=['Close'])
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False, header=['Close'])

print("Datasets saved successfully.")
