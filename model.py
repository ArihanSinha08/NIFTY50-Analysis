import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd

X_train = pd.read_csv('^X_train.csv')

# Define the architecture of the neural network
model = Sequential([
    # Input layer
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Dropout for regularization

    # Hidden layers
    Dense(32, activation='relu'),
    Dropout(0.2),

    # Output layer
    Dense(1, activation='linear')  # Linear activation for regression task
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print the model summary
print(model.summary())
