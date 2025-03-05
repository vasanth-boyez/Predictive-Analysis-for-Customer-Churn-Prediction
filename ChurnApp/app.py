import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title("📊 Customer Churn Prediction using CNN + LSTM")

# Load dataset only once
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.fillna(0, inplace=True)

    label_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
        "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
        "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
        "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
        "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
        "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3,
        },
        "Churn": {"No": 0, "Yes": 1},
    }

    for col, mapping in label_mappings.items():
        df[col] = df[col].map(mapping)

    X = df.drop("Churn", axis=1).values
    y = df["Churn"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")  # Save scaler for later use
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for CNN + LSTM

    return X, y, scaler, df, label_mappings

X, y, scaler, df, label_mappings = load_and_preprocess_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to build model
def build_model():
    model = Sequential([
        Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),

        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),

        Dense(32, activation='relu'),
        Dropout(0.4),

        Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model only if it is not already saved
if not os.path.exists("churn_prediction_model.h5"):
    st.info("Training the model for the first time. This may take a few minutes...")
    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    st.success(f"✅ Model trained! Test Accuracy: {accuracy:.4f}")
    
    model.save("churn_prediction_model.h5")
    st.info("Model saved as churn_prediction_model.h5")
else:
    model = tf.keras.models.load_model("churn_prediction_model.h5")
    st.success("✅ Model loaded successfully!")

# User input for prediction
st.subheader("🔍 Predict Customer Churn")
st.write("Enter customer details to predict whether they will churn or not.")

customer_data = []
for col in df.drop("Churn", axis=1).columns:
    if col in label_mappings:
        options = {v: k for k, v in label_mappings[col].items()}  # Reverse mapping
        selected_value = st.selectbox(f"{col} ({', '.join([f'{v}: {k}' for k, v in label_mappings[col].items()])})", list(options.keys()))
        customer_data.append(selected_value)
    else:
        value = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
        customer_data.append(value)

# Convert input to model format
customer_array = np.array(customer_data).reshape(1, -1)
scaler = joblib.load("scaler.pkl")  # Load saved scaler
customer_array = scaler.transform(customer_array).reshape(1, 1, len(customer_data))

if st.button("Predict"):
    prediction = model.predict(customer_array)
    result = np.argmax(prediction, axis=1)[0]  # 0: Not Churned, 1: Churned
    
    if result == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is likely to stay!")

st.write("📌 **Note:** Predictions are based on the pre-trained CNN + LSTM model.")
