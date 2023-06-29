import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')
import streamlit as st
import pyttsx3
import speech_recognition as sr
import time 
import os
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------ GET DATA -------------------

# %pip install meteostat --q
from meteostat import Stations, Daily
# Using the Meteostat API, we need the station ID of each state in Nigeria and the data start and ending dates

# Get the station IDs for each state in Nigeria
# Connect the API with the station ID and the start/end date
# Use 2021 as the start date and today's date as the end date

from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')

def get_data(state):
    station = Stations()
    nig_stations = station.region('NG')  # Filter stations by country code (NG for Nigeria)
    nig_stations = nig_stations.fetch()  # Fetch the station information
    global available_states
    # Some state names have a '/' in them, so we clean them up
    nig_stations['name'] = nig_stations['name'].apply(lambda x: x.split('/', 1)[0])
    nig_stations.drop_duplicates(subset=['name'], keep='first', inplace=True)
    available_states = nig_stations.name
    # Collect the data from the mentioned state if it is in the list of available states 
    try:
        state_stations = nig_stations[nig_stations['name'].str.contains(state)]
        station_id = state_stations.index[0]
    except IndexError:
        error = f'Sorry, {state} is not among the available states. Please choose another neighboring state'
        raise ValueError(error)

    # Connect to the API and fetch the data 
    data = Daily(station_id, str(state_stations.hourly_start[0]).split(' ')[0], str(current_date))
    data = data.fetch()

    # Collect the necessary features we might need 
    data['avg_temp'] = data[['tmin', 'tmax']].mean(axis=1)
    temp = data['avg_temp']
    press = data['pres']
    wind_speed = data['wspd']
    precip = data['prcp']
    rain_df = pd.concat([temp, press, wind_speed, precip], axis=1)

    # From the collected data, create a DataFrame for training the model 
    # Light rain — when the precipitation rate is < 2.5 mm (0.098 in) per hour. 
    # Moderate rain — when the precipitation rate is between 2.5 mm (0.098 in) – 7.6 mm (0.30 in) or 10 mm (0.39 in) per hour. 
    # Heavy rain — when the precipitation rate is > 7.6 mm (0.30 in) per hour, or between 10 mm (0.39 in) and 50 mm (2.0 in)
   
    rainfall = []
    for i in rain_df.prcp:
        if i < 0.1:
         rainfall.append('Light Rain')
        elif 0.1 <= i < 2.5:
         rainfall.append('Moderate Rain')
        else:
         rainfall.append('Heavy Rain')

    rain_df['Rainfall'] = rainfall

# Drop rows with missing values
    rain_df.dropna(inplace=True)

# Encode the categorical target variable
label_encoder = LabelEncoder()
rain_df['Rainfall_Code'] = label_encoder.fit_transform(rain_df['Rainfall'])

# Split the data into features and target variable
X = rain_df[['avg_temp', 'pres', 'wspd']]
y = rain_df['Rainfall_Code']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the XGBoost classifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# ------------------------------ VOICE RECOGNITION -------------------

# Initialize the speech recognition engine
r = sr.Recognizer()
mic = sr.Microphone()

# Function to convert text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to convert speech to text
def listen_to_speech():
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return ""

# ------------------------------ MAIN PROGRAM -------------------

def main():
    st.title("Weather Analysis and Rain Prediction")
    st.image('pngwing.com (5).png')
    st.write("Please speak the name of a state in Nigeria to get the weather information and rain prediction.")

    state = None
    paused = False
    stop = False

    while True:
        if stop:
            break
        if not state:
            st.write("Please speak the name of a state:")
            state = listen_to_speech()
            if state:
                state = state.lower()
                if state == "stop":
                    stop = True
                else:
                    if state not in available_states:
                        st.write(f"Sorry, {state} is not among the available states. Please try again.")
                        state = None
                    else:
                        st.write(f"Fetching weather data for {state}...")
                        get_data(state)
        else:
            st.write("Analyzing weather data and predicting rain...")
            # Perform the weather analysis and rain prediction here
            # ...

            st.write("Rain prediction complete.")
            st.write("Do you want to check the weather for another state?")
            speak_text("Do you want to check the weather for another state?")
            response = listen_to_speech()
            if response:
                response = response.lower()
                if response == "pause":
                    paused = True
                    state = None
                elif response == "stop":
                    stop = True
                else:
                    state = None
            else:
                state = None

        if paused:
            st.write("The program is paused. Please say 'resume' to continue or 'stop' to exit.")
            speak_text("The program is paused. Please say 'resume' to continue or 'stop' to exit.")
            response = listen_to_speech()
            if response:
                response = response.lower()
                if response == "resume":
                    paused = False
                elif response == "stop":
                    stop = True

if __name__ == "__main__":
    main()

