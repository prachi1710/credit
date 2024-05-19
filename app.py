import os
import sys
import platform
import numpy as np
import pandas as pd
import time
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load the model and encoders from files
loaded_model = XGBClassifier()
loaded_model.load_model("credit_score_multi_class_xgboost_model.json")
loaded_le = pickle.load(open("credit_score_multi_class_le.pkl", "rb"))
loaded_enc = pickle.load(open("credit_score_multi_class_ord_encoder.pkl", "rb"))
cat = ['Credit_Mix']

st.markdown("<h2 style='text-align:center; color:Orange;'>Credit Score Predictor</h2>", unsafe_allow_html=True)

# Define User Input Function
def user_input_data():
    # Sidebar Configuration
    Monthly_Inhand_Salary = st.sidebar.slider('Monthly_Inhand_Salary', 0.0, 100000.0, 25000.0, 1000.0)
    Total_EMI_per_month = st.sidebar.slider('Total_EMI_per_month', 0.0, 1780.0, 107.0, 0.1)
    Num_of_Delayed_Payment = st.sidebar.slider('Num_of_Delayed_Payment', 0, 25, 14, 1) 
    Delay_from_due_date = st.sidebar.slider('Delay_from_due_date', 0, 62, 21, 1)
    Changed_Credit_Limit = st.sidebar.slider('Changed_Credit_Limit', 0.5, 30.0, 9.40, 0.1)
    Num_Credit_Card = st.sidebar.slider('Num_Credit_Card', 0, 11, 5, 1)
    Outstanding_Debt = st.sidebar.slider('Outstanding_Debt', 0.0, 5000.0, 1426.0, 0.1)
    Interest_Rate = st.sidebar.slider('Interest_Rate', 1, 34, 14, 1)   
    Credit_Mix = st.sidebar.selectbox('Credit_Mix:', ['Standard', 'Bad', 'Good'])
    
    html_temp = """
    <div style="background-color:tomato;padding:1.5px">
    <h1 style="color:white;text-align:center;">Single Customer </h1>
    </div><br>"""
    st.sidebar.markdown(html_temp,unsafe_allow_html=True)
    
    data = { 
        'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Total_EMI_per_month': Total_EMI_per_month,
        'Num_of_Delayed_Payment': Num_of_Delayed_Payment, 
        'Delay_from_due_date': Delay_from_due_date,
        'Changed_Credit_Limit': Changed_Credit_Limit,
        'Num_Credit_Card': Num_Credit_Card,        
        'Outstanding_Debt': Outstanding_Debt,
        'Interest_Rate': Interest_Rate,       
        'Credit_Mix': Credit_Mix,
    }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data

# Sidebar Configuration
st.sidebar.header("User input parameter")
df = user_input_data() 

col1, col2 = st.columns([4, 6])

with col1:
    if st.checkbox('Show User Inputs:', value=True):
        st.dataframe(df.astype(str).T.rename(columns={0:'input_data'}).style.highlight_max(axis=0))

with col2:
    for i in range(2): 
        st.markdown('#')
    if st.button('Make Prediction'):   
        sound = st.empty()
        video_html = """
            <iframe width="0" height="0" 
            src="https://www.youtube-nocookie.com/embed/t3217H8JppI?rel=0&amp;autoplay=1&mute=0&start=2860&amp;end=2866&controls=0&showinfo=0" 
            allow="autoplay;"></iframe>
            """
        sound.markdown(video_html, unsafe_allow_html=True)   
        
        # Use the loaded model for predictions
        df[cat] = loaded_enc.transform(df[cat]) 
        prediction = loaded_model.predict(df)
        prediction = loaded_le.inverse_transform(prediction)[0]

        time.sleep(3.7)  # wait for 2 seconds to finish the playing of the audio
        sound.empty()  # optionally delete the element afterwards   
        
        st.success(f'Credit score probability is:&emsp;{prediction}')
