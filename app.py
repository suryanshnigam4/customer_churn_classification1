import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model
model = tf.keras.models.load_model("customer_churn_model.keras")

# Load preprocessing objects
le = joblib.load("label_encoder.pkl")
ct = joblib.load("encoder.pkl")
sc = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

# Input widgets
credit_score = st.slider("Credit Score", 300, 900, 600)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider("Age", 18, 100, 30)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.slider(
    "Balance",
    0.0,
    250000.0,
    50000.0
)

products = st.slider(
    "Number of Products",
    1,
    4,
    2
)

card = st.selectbox(
    "Has Credit Card",
    [0,1]
)

active = st.selectbox(
    "Is Active Member",
    [0,1]
)

salary = st.slider(
    "Estimated Salary",
    0.0,
    200000.0,
    50000.0
)

if st.button("Predict"):

    customer = np.array([[credit_score,
                          geography,
                          gender,
                          age,
                          tenure,
                          balance,
                          products,
                          card,
                          active,
                          salary]])

    customer[:,2] = le.transform(customer[:,2])

    customer = ct.transform(customer)

    customer = sc.transform(customer)

    prediction = model.predict(customer)

    if prediction[0][0] >= 0.5:
        st.error("Customer is likely to leave the bank.")
    else:
        st.success("Customer is likely to stay with the bank.")

    st.write("Probability:", round(float(prediction[0][0]),2))