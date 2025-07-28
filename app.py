import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load(r"C:\Users\rabad\OneDrive\Desktop\New folder\ready\gradient_boosting_model.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# App title
st.title("💼 Employee Salary Prediction App")
st.markdown("This app predicts employee salaries based on input features using a trained Gradient Boosting model.")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Employee Details")

    # Example input fields (adjust based on your dataset)
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    education = st.selectbox("Education", ["Bachelors", "Masters", "PhD", "HS-grad", "Some-college", "Others"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales",
                                              "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                                              "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                                              "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)

    submitted = st.form_submit_button("Predict Salary")

# Handle prediction
if submitted:
    input_df = pd.DataFrame({
        "age": [age],
        "education": [education],
        "occupation": [occupation],
        "hours_per_week": [hours_per_week],
        "race": [race],
        "sex": [gender],
        "capital_gain": [capital_gain],
        "capital_loss": [capital_loss]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: **${round(prediction, 2)}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")