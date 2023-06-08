import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Preprocessing the data
# Load the data from the CSV file
data = pd.read_csv('simulated_data.csv')

# Split the data into features (x) and output labels (y)
X = data.drop(columns=['Currency Pair'])
Y = data['Price']

# Perform any preprocessing steps (e.g normalization)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Apply normalization to input features
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define a custom profit metric function
def profit_metric(y_true, y_pred):
    # Calculate the percentage retur based on the predicted and true prices
    return tf.reduce_mean((y_pred - y_true) / y_true) * 100.0

# Buildig the Neural Network
model = tf.keras.models.Sequential()

# Define the layers and activation functions of the nn
model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(1,)))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[profit_metric])

# Training the Neuran Network
model.fit(X_train_normalized, Y_train, epochs=100, batch_size=32, validation_data=(X_test_normalized, Y_test))

# Evaluate the trained model o the testing data
loss, profit = model.evaluate(X_test_normalized, Y_test)

# Make predictions using the trained model
predictions = model.predict(X_test_normalized)

