
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = pickle.load(open("rfc.pkl", "rb"))

# Create the Streamlit app
st.title("Employee Churn Prediction")

# Input features with default values and corrected feature list
features = {
    'department': ["sales", "product", "marketing", "technology", "support", "engineering", "management", "information_technology", "hr", "accounting", "finance", "procurement"],
    'salary': ["low", "medium", "high"],
    'filed_complaint': 0.0,
    'last_evaluation': 0.0,
    'recently_promoted': 0.0,
    'satisfaction': 0.0,
    'tenure': 0.0,
    'avg_monthly_hrs': 0,
    'n_projects': 0,
    'n_companies': 0, # Added missing feature
    'years_experience':0, # Added missing feature
    'is_manager': 0.0, # Added missing feature
    'status': 0.0 # Added missing feature
     # Add all 22 features here with defaults
}

input_data = {}
for feature, default_value in features.items():
  if isinstance(default_value, list):
    input_data[feature] = st.selectbox(feature.capitalize(), default_value)
  elif isinstance(default_value, float):
    input_data[feature] = st.number_input(feature.capitalize(), min_value=0.0, max_value=1.0, value=default_value)
  elif isinstance(default_value, int):
    input_data[feature] = st.number_input(feature.capitalize(), min_value=0, value=default_value)

# Create a DataFrame for the input features
input_df = pd.DataFrame([input_data])

# Preprocess the input data using the same ColumnTransformer as during training
# Note: We are re-creating the transformation here, ensuring consistency
# IMPORTANT:  The categories in the OneHotEncoder MUST match the training data
categorical_features = ['department', 'salary']
ct = ColumnTransformer([("trf1", OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_features)], remainder="passthrough")

# Get feature names after one-hot encoding (important for consistent order)
feature_names = list(ct.fit(pd.DataFrame(features,index=[0])).get_feature_names_out())

input_data_encoded = ct.transform(input_df)

# Check if the number of features matches the model's expectation
if input_data_encoded.shape[1] != 22:
    st.error(f"Error: Input features should be 22, but got {input_data_encoded.shape[1]}. Please provide all 22 features with valid values.")
else:
    # Make the prediction
    prediction = model.predict(input_data_encoded)

    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: The employee is likely to stay.")
    else:
        st.write("Prediction: The employee is likely to leave.")
