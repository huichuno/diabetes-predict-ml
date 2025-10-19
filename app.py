import streamlit as st
import pickle
import pandas as pd
import os
from PIL import Image

def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, model_type):
    """
    Make diabetes predictions using the trained models
    
    Parameters:
    - gender: 0 = Female, 1 = Male
    - age: Age in years
    - hypertension: 0 = No, 1 = Yes
    - heart_disease: 0 = No, 1 = Yes  
    - smoking_history: 0-5 ['No Info', 'current', 'ever', 'former', 'never', 'not current']
    - bmi: Body Mass Index
    - hba1c_level: HbA1c level
    - blood_glucose_level: Blood glucose level
    - model_type: Logistic Regression, Random Forest
    
    Returns:
    - prediction: 0 = No Diabetes, 1 = Diabetes
    - probability: [prob_no_diabetes, prob_diabetes]
    """    
    # Feature names in the correct order
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    # Create DataFrame with proper feature names
    patient_data = pd.DataFrame([[0 if gender == "Female" else 1,
                                  age,
                                  0 if hypertension == "No" else 1,
                                  0 if heart_disease == "No" else 1,
                                  0 if smoking_history == "No Info" else 1 if smoking_history == "current" else 2 if smoking_history == "ever" else 3 if smoking_history == "former" else 4 if smoking_history == "never" else 5,
                                  bmi,
                                  hba1c_level,
                                  blood_glucose_level]],
                                columns=feature_names)
    
        # Choose model
    if model_type == 'Logistic Regression':
        # Load Logistic Regression model
        with open('bin/diabetes_prediction_model_lr.bin', 'rb') as fp:
            model = pickle.load(fp)
    elif model_type == 'Random Forest':
        # Load Random Forest model
        with open('bin/diabetes_prediction_model_rf.bin', 'rb') as fp:
            model = pickle.load(fp)
    else:
        raise ValueError("Invalid model type. Choose 'Logistic Regression' or 'Random Forest'.")

    # Make prediction
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0]

    return prediction, probability


st.set_page_config(layout="wide")
st.sidebar.title("Machine Learning Models")
st.sidebar.write("Binary Classification:")
model_option = st.sidebar.selectbox("Select model:", ["Select Model", "Logistic Regression", "Random Forest"], index=0)

try:
    current_directory = os.getcwd()
    image_path = os.path.join(current_directory, "assets", "healthcare.png")
    im = Image.open(image_path)
    st.image(im, width=90)

    st.title("Diabetes Xpert") 

    st.markdown("""
    We are a smart and user-friendly application that predicts the \
    likelihood of patient having diabetes based on information they provide. \
    We uses advanced algorithm to analyze key health data, helping users and healthcare professionals get an early indication of diabetes risk for timely intervention and improved health management.
    """)

    with st.sidebar.container():
        if model_option == "Logistic Regression":
            with st.sidebar:
                st.write("Accuracy: 96.04%")
                st.write("Precision: 87.41%")
                st.write("Recall: 62.89%")
                st.write("F1 Score: 73.15%")

                if os.path.isfile('bin/diabetes_prediction_model_lr.bin'):
                    st.sidebar.success(f"{model_option} model selected!")
                else:
                    st.sidebar.error(f"{model_option} model file not found!")
                    raise ValueError(f"{model_option} model file not found!")

        elif model_option == "Random Forest":
            with st.sidebar:
                st.write("Accuracy: 97.04%")
                st.write("Precision: 95.16%")
                st.write("Recall: 68.91%")
                st.write("F1 Score: 79.94%")

                if os.path.isfile('bin/diabetes_prediction_model_rf.bin'):
                    st.sidebar.success(f"{model_option} model selected!")
                else:
                    st.sidebar.error(f"{model_option} model file not found!")
                    raise ValueError(f"{model_option} model file not found!")

        else:
            st.sidebar.write("Select a model to see its description.")        

    if model_option != "Select Model":    
        # user input form
        st.subheader("Patient Information")
        
        with st.form("personal_info", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                HbA1c_level = st.slider("Blood HbA1c Level", min_value=3.0, max_value=9.0, value=3.0, step=0.1)
                blood_glucose_level = st.slider("Blood Glucose Level", min_value=50, max_value=300, value=50, step=1)
                bmi = st.slider("BMI: Weight (kg) / Height (m)^2", min_value=10.0, max_value=100.0, value=10.0, step=0.1)
                age = st.slider("Age", min_value=1, max_value=120, value=1, step=1)
            with col2:
                smoking_history = st.selectbox("Smoking History:", ['', 'No Info', 'Current', 'Former', 'Never'], index=4)
                hypertension = st.selectbox("Hypertension:", ['', 'No', 'Yes'], index=1)
                heart_disease = st.selectbox("Heart Disease:", ['', 'No', 'Yes'], index=1)
                gender = st.selectbox("Gender:", ['', 'Female', 'Male'], index=0)

            submit = st.form_submit_button(label="Submit", icon="🔄")
            if submit:
                if (HbA1c_level is None or blood_glucose_level is None or bmi is None or age is None or
                    smoking_history == '' or hypertension == '' or heart_disease == '' or gender == ''):
                    st.warning("Please fill in all fields.")
                else:            
                    prediction, probability = predict_diabetes(gender=gender,
                                                            age=age, hypertension=hypertension,
                                                            heart_disease=heart_disease, 
                                                            smoking_history=smoking_history,
                                                            bmi=bmi,
                                                            hba1c_level=HbA1c_level,
                                                            blood_glucose_level=blood_glucose_level,
                                                            model_type=model_option)
                    
                    if prediction == 1:
                        st.info(f"Details: Gender:{gender}, Age:{age}, HTN:{hypertension}, HD:{heart_disease}, Smoke:{smoking_history}, BMI:{bmi}, HbA1c:{HbA1c_level}, Glucose:{blood_glucose_level}")
                        st.error(f"{model_option} Prediction: The patient is LIKELY to have diabetes. (Confidence: {max(probability) * 100:.3f}%)")
                    else:
                        st.info(f"Details: Gender:{gender}, Age:{age}, HTN:{hypertension}, HD:{heart_disease}, Smoke:{smoking_history}, BMI:{bmi}, HbA1c:{HbA1c_level}, Glucose:{blood_glucose_level}")
                        st.success(f"{model_option} Prediction: The patient is UNLIKELY to have diabetes. (Confidence: {max(probability) * 100:.3f}%)")

    else:
        st.info("Please select a model from the sidebar to make predictions.")

except Exception as e:
    st.error(e)
