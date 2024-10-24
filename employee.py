import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and ColumnTransformer
model = pickle.load(open("rfc.pkl", "rb"))


# Create the Streamlit app
st.title("Employee Churn Prediction")

# Input features
department = st.selectbox("Department", ["sales", "product", "marketing", "technology", "support", 
                                          "engineering", "management", "information_technology", 
                                          "hr", "accounting", "finance", "procurement"])
salary = st.selectbox("Salary", ["low", "medium", "high"])

filed_complaint = st.number_input("Filed Complaint", min_value=0.0, max_value=1.0)
last_evaluation = st.number_input("Last Evaluation Score", min_value=0.0, max_value=1.0)
recently_promoted = st.number_input("Recently Promoted", min_value=0.0, max_value=1.0)
satisfaction = st.number_input("Satisfaction", min_value=0.0, max_value=1.0)
tenure = st.number_input("Tenure", min_value=0.0)
avg_monthly_hrs = st.number_input("Average Monthly Hours", min_value=0)
n_projects = st.number_input("Number of Projects", min_value=0)

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'department': [department],
    'salary': [salary],
    'filed_complaint': [filed_complaint],
    'last_evaluation': [last_evaluation],
    'recently_promoted': [recently_promoted],
    'satisfaction': [satisfaction],
    'tenure': [tenure],
    'avg_monthly_hrs': [avg_monthly_hrs],
    'n_projects': [n_projects]
})

# Preprocess the input data using the loaded ColumnTransformer
input_data_encoded = ct.transform(input_data)

# Make the prediction
prediction = model.predict(input_data_encoded)

# Display the prediction
if prediction[0] == 0:
    st.write("Prediction: The employee is likely to stay.")
else:
    st.write("Prediction: The employee is likely to leave.")
