import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert 'TotalCharges' to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.fillna(0, inplace=True)  # Fill missing values with 0

# Encode categorical variables
label_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", 
              "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
              "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]

for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Select features and target
X = df.drop("Churn", axis=1).values
y = df["Churn"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape input for CNN + LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])  # Use time_steps=1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build robust CNN + LSTM model
model = Sequential([
    Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=1),
    
    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    LSTM(32, dropout=0.3, recurrent_dropout=0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.4),
    
    Dense(2, activation='softmax')  # Output as binary classification
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
model.save("churn_prediction_model.h5")
print("Model saved as churn_prediction_model.h5")

# Make predictions and convert to binary labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Display a few predictions
print("Predictions (0: Not Churned, 1: Churned):", y_pred_labels[:10])
