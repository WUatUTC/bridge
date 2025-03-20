# This app is used to deploy the web app for lab_11
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Load the saved preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load the TensorFlow model
model = tf.keras.models.load_model("tf_bridge_model.h5")

st.title("Bridge Load Capacity Prediction")

# User inputs
span = st.number_input("Bridge Span (ft)", value=250.0)
deck_width = st.number_input("Deck Width (ft)", value=40.0)
age = st.number_input("Age (years)", value=20)
num_lanes = st.number_input("Number of Lanes", value=4)
material = st.selectbox("Material", options=["Steel", "Concrete", "Composite"])
condition = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=4)

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Span_ft': [span],
    'Deck_Width_ft': [deck_width],
    'Age_Years': [age],
    'Num_Lanes': [num_lanes],
    'Condition_Rating': [condition],
    'Material': [material]
})

if st.button("Predict Load Capacity"):
    # Transform input using the preprocessor
    X_input = preprocessor.transform(input_data)
    prediction = model.predict(np.array(X_input, dtype=np.float32))
    st.write(f"Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")
