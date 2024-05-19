
import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# Load trained models
loaded_ord_enc = pickle.load(open("credit_score_multi_class_ord_encoder.pkl", "rb"))
loaded_le = pickle.load(open("credit_score_multi_class_le.pkl", "rb"))
loaded_dummy_enc = pickle.load(open("credit_score_multi_class_dummy.pkl", "rb"))

# Load XGBoost model
loaded_model = xgb.Booster()
loaded_model.load_model("credit_score_multi_class_xgboost_model.json")

# Function to preprocess input data
def preprocess_input(data):
    # Perform any necessary preprocessing (e.g., encoding categorical features)
    # Here, you would use the loaded encoders to transform the input data
    #le.inverse processed_data
    return processed_data

# Function to make predictions
def predict_credit_score(data):
    # Preprocess input data
    processed_data = preprocess_input(data)
    
    # Make predictions
    predictions = loaded_model.predict(processed_data)
    
    return predictions

# Streamlit app code (same as before)
def main():
    st.title("Credit Score Prediction App")
    st.write("Enter customer information to predict their credit score.")

    # Create input fields
    total_emi = st.number_input("Total EMI per month", min_value=0)
    num_bank_accounts = st.number_input("Number of bank accounts", min_value=0)
    # Add more input fields for other features

    # Create a dictionary from user inputs
    user_data = {
        "Total_EMI_per_month": total_emi,
        "Num_Bank_Accounts": num_bank_accounts,
        # Add more key-value pairs for other features
    }

    # When the user clicks the predict button
    if st.button("Predict Credit Score"):
        # Convert the dictionary into a DataFrame
        input_df = pd.DataFrame([user_data])

        # Predict credit score
        prediction = predict_credit_score(input_df)

        # Display prediction
        st.write("Predicted Credit Score:", prediction)

# Run the app
if __name__ == "__main__":
    main()