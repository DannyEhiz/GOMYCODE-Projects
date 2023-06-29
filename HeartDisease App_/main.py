import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
import joblib


# load the model
model = joblib.load(open('LinReg.pkl', 'rb'))

st.markdown("<h1 style = 'text-align: right; color: #F9D949'>HEART DISEASE TESTER</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: right; color: #FFB4B4'>Built by GoMyCode Scions</h6>", unsafe_allow_html = True)

st.image('pngwing.com (29).png')
# st.title('HEART RATE TESTER')
# st.subheader('Built By Gomycode Scions')

# img1 = st.image('images\pngwing.com (21).png')

st.write('Pls register your name for record of usage')
username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")


st.sidebar.image('pngwing.com (24).png', caption = username, use_column_width = True)

input_type = st.sidebar.selectbox('Choose your preferred input type',['Number Input', 'Slider'])

if input_type == 'Number Input':
    biking = st.sidebar.number_input('Biking', 1.1 , 75.0, 38.0 )
    smoking = st.sidebar.number_input('Smoking', 0.5, 30.0, 15.4 )
else:
    biking = st.sidebar.slider('Biking', 1.1 , 75.0, 38.0 )
    smoking = st.sidebar.slider('Smoking', 0.5, 30.0, 15.4 )

# Aggregate the Input Vlaues for our test

input_values = [[biking, smoking]]
frame = ({'biking': [biking],
          'smoking': [smoking]
        }) 

st.markdown('<hr>', unsafe_allow_html = True)
st.write('These are your input variables')
frame = pd.DataFrame(frame)
frame = frame.rename(index = {0: 'Value'})
frame = frame.transpose()
st.write(frame)

# Testing the model 
prediction = model.predict(input_values)
st.success(prediction)
if prediction > 10:
    st.error('You are risk of having a heart attack')
    st.image('pngwing.com (30).png', width = 300)
else:
    st.success('Your heart is healthy')
    st.image('pngwing.com (31).png')






