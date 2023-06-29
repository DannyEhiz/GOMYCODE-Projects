import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

data = pd.read_csv('iris.data.csv', header = None)

data.rename(columns = {0: 'sepal length (cm)', 1: 'sepal width (cm)', 2: 'petal length (cm)', 3:  'petal width (cm)', 4: 'names'}, inplace = True)

print(data.head())

x = data.drop('names', axis = 1)
y = data.names

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into train and test
x_train , x_test , y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# create dataframe for train data and test data.
train_data = pd.concat([x_train, pd.Series(y_train)], axis = 1)
test_data = pd.concat([x_test, pd.Series(y_test)], axis = 1)

# Model Creation
logReg = RandomForestClassifier()
logRegFitted = logReg.fit(x_train, y_train)
y_pred = logRegFitted.predict(x_test)

# acc = logReg.score(y_pred, y)

heatmap = sns.heatmap(data.corr(), cmap = 'BuPu', annot = True )

# Save the Model
import joblib
joblib.dump(logReg, 'Logistic_Model.pkl')

# print(acc)

# ------------------------------------ START WITH STREAMLIT IMPLEMENTATION -----------------------------------------------------
st.markdown("<h1 style = 'text-align: right; color: #D25380'>IRIS PREDICTOR APP</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: right; color: #FFB4B4'>Built by GoMyCode Scions</h6>", unsafe_allow_html = True)

img1 = st.image('images\pngwing.com (28).png')

st.write('Pls register your name for record of usage')
username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")



st.sidebar.subheader(f"Hey {username}")
metric = st.sidebar.radio('How do you want your feature input?\n \n \n', ('slider', 'direct input'))


if metric == 'slider':
   sepal_length = st.sidebar.slider('SEPAL LENGTH', 0.0, 9.0, (5.0))

   sepal_width = st.sidebar.slider('SEPAL WIDTH', 0.0, 4.5, (2.5))

   petal_length = st.sidebar.slider('PETAL LENGTH', 0.0, 8.0, (4.5))

   petal_width = st.sidebar.slider('PETAL WIDTH', 0.0, 3.0, (1.5))
else:
    sepal_length = st.sidebar.number_input('SEPAL LENGTH')
    sepal_width = st.sidebar.number_input('SEPAL WIDTH')
    petal_length = st.sidebar.number_input('PETAL LENGTH')
    petal_width = st.sidebar.number_input('PETAL WIDTH')

st.write('Selected Inputs: ', [sepal_length, sepal_width, petal_length, petal_width])

input_values = [[sepal_length, sepal_width, petal_length, petal_width]]

# import the model
model = joblib.load(open('Logistic_Model.pkl', 'rb'))
prediction = model.predict(input_values)



if prediction == 0:
    st.success('The Flower is an Iris-setosa')
    setosa = Image.open('images\Irissetosa1.JPG')
    st.image(setosa, caption = 'Iris-setosa', width = 400)
elif pred == 1:
    st.success('The Flower is an Iris-versicolor ')
    versicolor = Image.open('images\irisversicolor.JPG')
    st.image(versicolor, caption = 'Iris-versicolor', width = 400)
else:
    st.success('The Flower is an Iris-virginica')
    st.image(virginica, caption = 'Iris-virginica', width = 400  ')
    virginica = Image.open('images\Iris-virginica.JPG'))

