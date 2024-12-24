import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the trained model, scaler, and label encoders
model = pickle.load(open("model.pkl", "rb"))  # Update with your model file path
scaler = pickle.load(open("scaler.pkl", "rb"))  # Update with your scaler file path

# Define categorical columns
cat_cols = ['University_rating', 'SOP', 'LOR', 'Research']

# Load label encoders
label_encoders = {}
for col in cat_cols:
    with open(f"{col}_label_encoder.pkl", "rb") as f:
        label_encoders[col] = pickle.load(f)

# Title
st.title("Graduate Admission Prediction App")

# Input fields
st.header("Enter Applicant Details")
GRE_score = st.number_input("GRE Score", min_value=260, max_value=340, step=1)
TOEFL_score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
CGPA = st.number_input("CGPA (on a scale of 10)", min_value=0.0, max_value=10.0, step=0.1)

# Input fields for categorical columns with sliders
University_rating = st.slider("University Rating (0 to 5)", min_value=0.0, max_value=5.0, step=0.5)
SOP = st.slider("Statement of Purpose Strength (0 to 5)", min_value=0.0, max_value=5.0, step=0.5)
LOR = st.slider("Letter of Recommendation Strength (0 to 5)", min_value=0.0, max_value=5.0, step=0.5)
Research = st.selectbox("Research Experience", options=["No", "Yes"])

# Process user inputs
if st.button("Predict"):
    # Create a DataFrame from inputs
    user_input = pd.DataFrame({
        'GRE_score': [GRE_score],
        'TOEFL_score': [TOEFL_score],
        'University_rating': [University_rating],
        'SOP': [SOP],
        'LOR': [LOR],
        'CGPA': [CGPA],
        'Research': [Research]
    })

    # Encode categorical columns
    for col in cat_cols:
        # Handle unseen labels in categorical columns
        user_input[col] = user_input[col].apply(
            lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0
        )

    # Reorder the columns to match the training data (ensure the same order as when the scaler was fitted)
    input_columns = ['GRE_score', 'TOEFL_score', 'University_rating', 'SOP', 'LOR', 'CGPA', 'Research']
    user_input = user_input[input_columns]

    # Scale the data
    scaled_features = scaler.transform(user_input)

    # Make predictions
    prediction = model.predict(scaled_features)
    prediction= prediction*100
    #probability = model.predict_proba(scaled_features)[0][1]  # Confidence for positive class

    # Display the result
    st.subheader("Prediction Result")
    st.success(f"The predicted admission score is: {prediction[0]:.2f}%")