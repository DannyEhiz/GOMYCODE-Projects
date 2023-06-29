
import streamlit as st 
import pandas as pd
import numpy as np
# %pip install matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')



# import some data to play with
data = pd.read_csv('iris.data.csv', header = None)

data.rename(columns = {0: 'sepal length (cm)', 1: 'sepal width (cm)', 2: 'petal length (cm)', 3:  'petal width (cm)', 4: 'names'}, inplace = True)

x = data.drop(['names'], axis = 1)
y = data['names']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into train and test
x_train , x_test , y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# create dataframe for train data and test data.
train_data = pd.concat([x_train, pd.Series(y_train)], axis = 1)
test_data = pd.concat([x_test, pd.Series(y_test)], axis = 1)

# Model Creation
logReg = LogisticRegression()
logRegFitted = logReg.fit(x_train, y_train)
y_pred = logRegFitted.predict(x_test)

# using R2 score module from sklearn metrics for the goodness to fit information
score = r2_score(y_test,y_pred)
print(f"the score is {score}, Good job")
print(data.head())

print(f"{score} is the accuracy score")
# saving the model using joblib
import joblib
joblib.dump(logReg, 'Logistic_Model.pkl')

# -----------------------------------------------------
# FROM HERE WE BEGIN THE IMPLEMENTATION FOR STREAMLIT.

st.header('IRIS MODEL DEPLOYMENT')
user_name = st.text_input('Register User')

if(st.button('SUBMIT')):
    st.write(f"You are welcome {user_name}. Enjoy your usage")

st.write(data)

from PIL import Image
image = Image.open(r'images\use.png')
st.sidebar.image(image)

# // I want to connect to an sql server?

st.sidebar.subheader(f"Hey {user_name}")
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

# Modelling
# import the model
model = joblib.load(open('Logistic_Model.pkl', 'rb'))
pred = model.predict(input_values)


# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_test)
# st.pyplot(fig)


if pred == 0:
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