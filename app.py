import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Add enhanced custom CSS
st.markdown(
    """
    <style>
    /* General Body Styling */
    body {
        background: linear-gradient(to bottom, #e3f2fd, #f3e5f5);
        font-family: 'Roboto', sans-serif;
    }

    /* Title Styling */
    .stTitle {
        color: #2d3436;
        text-align: center;
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 30px;
    }

    /* Input Box Styling */
    div.stSelectbox, div.stSlider, div.stNumberInput {
        font-family: 'Roboto', sans-serif;
        margin-bottom: 20px;
    }

    /* Streamlit Buttons */
    button[kind="primary"] {
        background: linear-gradient(to right, #ff6f61, #ff9068);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px !important;
        font-size: 16px !important;
        transition: all 0.3s ease-in-out;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(to right, #ff9068, #ff6f61);
        transform: scale(1.05);
    }

    /* Churn Prediction Result */
    .prediction-result {
        font-size: 1.4rem;
        color: #2d3436;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        border-radius: 10px;
    }

    .churn-yes {
        background: linear-gradient(to right, #e74c3c, #e57373);
        color: white;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .churn-no {
        background: linear-gradient(to right, #27ae60, #66bb6a);
        color: white;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Input Label Styling */
    label {
        color: #2d3436;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Footer Styling */
    footer {
        text-align: center;
        font-family: 'Roboto', sans-serif;
        margin-top: 50px;
        color: #636e72;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app title
st.markdown("<h1 class='stTitle'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display churn prediction probability
st.markdown(
    f"<div class='prediction-result'>Churn Probability: <b>{prediction_proba:.2f}</b></div>",
    unsafe_allow_html=True
)

# Display churn prediction result
if prediction_proba > 0.5:
    st.markdown(
        "<div class='prediction-result churn-yes'>The customer is likely to churn.</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div class='prediction-result churn-no'>The customer is not likely to churn.</div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    "<footer>Made with ❤️ using Streamlit</footer>",
    unsafe_allow_html=True
)
