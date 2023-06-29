import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('all_perth_310121.csv')

data['DATE_SOLD'] = pd.to_datetime(data['DATE_SOLD'])

cat = data.select_dtypes(include = ['category', 'object'])
num = data.select_dtypes(include = 'number') 

# copy data
df = data.copy()

# clean according to datatype
for item in num:
    df[item].fillna(df[item].median(),inplace=True)

# 3. categorise the price column into bins
bins = [0,500000,700000,2550000] 
df['PRICE_GROUP'] = pd.cut(df['PRICE'], bins, labels=['low_cost', 'average','elite'])

# Reshare into data types
categorical = df.select_dtypes(include = ['category', 'object'])
numerical = df.select_dtypes(include = 'number')

# Encode and Scale
encoder = LabelEncoder()
for i in cat.columns:
    if i in df.columns:
        df[i] = encoder.fit_transform(df[i])

scaler = StandardScaler()
for i in num.columns:
    if i in df.columns:
        df[[i]] = scaler.fit_transform(df[[i]])

x_trans = df.drop(['PRICE_GROUP', 'DATE_SOLD', 'PRICE'], axis = 1)
y_trans = df.PRICE_GROUP

best_feature1 = SelectKBest(score_func = f_classif, k = 'all')
fitting1 = best_feature1.fit(x_trans,y_trans)
scores1 = pd.DataFrame(fitting1.scores_)
columns1 = pd.DataFrame(x_trans.columns)
feat_score1 = pd.concat([columns1, scores1], axis = 1)
feat_score1.columns = ['Feature', 'F_classif_score'] 
selected = feat_score1.nlargest(10, 'F_classif_score')
selected.sort_values(by = 'F_classif_score', ascending = False)

selected = ['FLOOR_AREA', 'NEAREST_SCH_RANK', 'BATHROOMS', 'CBD_DIST',
       'BEDROOMS', 'POSTCODE', 'LAND_AREA', 'GARAGE', 'LONGITUDE',
       'BUILD_YEAR']

new_feats = df[['FLOOR_AREA', 'NEAREST_SCH_RANK', 'BATHROOMS', 'CBD_DIST',
       'BEDROOMS', 'POSTCODE', 'LAND_AREA', 'GARAGE', 'LONGITUDE',
       'BUILD_YEAR']]

y_trans = encoder.fit_transform(y_trans)

x_train , x_test , y_train, y_test = train_test_split(new_feats, y_trans, test_size = 0.2, stratify = y_trans, random_state = 2)

print(f"\t this is the selected dataframe \n {new_feats.head()}")

# MODELLING

xgb_model = XGBClassifier()

xgb_model.fit(x_train, y_train)

# Cross-Validate
validate = xgb_model.predict(x_train)
print(f"\t \n \n \n Validation: \n {acc(y_train, validate)}")

prediction = xgb_model.predict(x_test)
print(f"\t \n \n \n Test: \n {acc(y_test, prediction)}")

import joblib
joblib.dump(xgb_model, 'house_model.pkl')


# -----------------------------STREAMLIT IMPLEMENTATION STARTS HERE---------------------

st.markdown("<h1 style = 'text-align: right; color: #F5C6EC'>ALL PERTH HOUSE PRICE APP</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; text-align: right; color: #FFB4B4'>Built by GoMyCode Pumpkin Reedemers</h6>", unsafe_allow_html = True)

st.image('image\pngwing.com (24).png', caption = "Pumpkin's House", width = 500)

user_name = st.text_input('Welcome. Pls Enter users name')
if st.button('Submit'):
    st.write(f'Welcome {user_name}')

side_img1 = st.sidebar.image('image\pngwing.com (25).png', caption = user_name, width = 300)

entry = st.sidebar.selectbox('How do you want to enter your variables', ['Direct Input', 'Slider']) 


[['FLOOR_AREA', 'NEAREST_SCH_RANK', 'BATHROOMS', 'CBD_DIST',
       'BEDROOMS', 'POSTCODE', 'LAND_AREA', 'GARAGE', 'LONGITUDE',
       'BUILD_YEAR']]

if entry == 'Direct Input':
    FLOOR_AREA = st.sidebar.number_input('FLOOR_AREA',min_value = 1, max_value = 870)
    NEAREST_SCH_RANK = st.sidebar.number_input('NEAREST_SCH_RANK',min_value = 1.0, max_value = 139.0)
    BATHROOMS = st.sidebar.number_input('BATHROOMS',min_value = 1, max_value = 16)
    CBD_DIST = st.sidebar.number_input('CBD_DIST',min_value = 681, max_value = 59800)
    BEDROOMS = st.sidebar.number_input('BEDROOMS',min_value = 0, max_value = 10)
    POSTCODE = st.sidebar.number_input('POSTCODE',min_value = 6003, max_value = 6558)
    LAND_AREA = st.sidebar.number_input('LAND_AREA',min_value = 61, max_value = 999999)
    GARAGE = st.sidebar.number_input('LAND_AREA',min_value = 1, max_value = 99)
    LONGITUDE = st.sidebar.number_input('LAND_AREA',min_value = 115.58273, max_value = 116.343201)
    BUILD_YEAR = st.sidebar.number_input('LAND_AREA',min_value = 1868, max_value = 2017)

elif entry == 'Slider':
    FLOOR_AREA = st.sidebar.number_input('FLOOR_AREA',min_value = 1, max_value = 870)
    NEAREST_SCH_RANK = st.sidebar.number_input('NEAREST_SCH_RANK',min_value = 1.0, max_value = 139.0)
    BATHROOMS = st.sidebar.number_input('BATHROOMS',min_value = 1, max_value = 16)
    CBD_DIST = st.sidebar.number_input('CBD_DIST',min_value = 681, max_value = 59800)
    BEDROOMS = st.sidebar.number_input('BEDROOMS',min_value = 0, max_value = 10)
    POSTCODE = st.sidebar.number_input('POSTCODE',min_value = 6003, max_value = 6558)
    LAND_AREA = st.sidebar.number_input('LAND_AREA',min_value = 61, max_value = 999999)
    GARAGE = st.sidebar.number_input('GARAGE',min_value = 1, max_value = 99)
    LONGITUDE = st.sidebar.number_input('LONGITUDE',min_value = 115.58273, max_value = 116.343201)
    BUILD_YEAR = st.sidebar.number_input('BUILD_YEAR',min_value = 1868, max_value = 2017)



# Define all input variables
input_variables = [[FLOOR_AREA, NEAREST_SCH_RANK, BATHROOMS, CBD_DIST, BEDROOMS, POSTCODE, LAND_AREA, GARAGE, LONGITUDE, BUILD_YEAR]]


frame = ({'FLOOR_AREA':[FLOOR_AREA], 
        'NEAREST_SCH_RANK':[NEAREST_SCH_RANK], 'BATHROOMS':[BATHROOMS], 'CBD_DIST':[CBD_DIST], 'BEDROOMS':[BEDROOMS], 'POSTCODE':[POSTCODE], 'LAND_AREA':[LAND_AREA], 'GARAGE': [GARAGE], 'LONGITUDE': [LONGITUDE], 'BUILD_YEAR': [BUILD_YEAR]
        })

frame = pd.DataFrame(frame)
st.write(frame)

frames = frame.copy()
frame = frame.rename(index = {0: 'Value'})
frame = frame.transpose()
frame[['Value']] = scaler.fit_transform(frame[['Value']])

holder = [[]]
# extract the transformed variables and append them to holder
for i in frame.values:
    holder.append(float(i))

# turn the holder to a numpy so it can be usable for the xgboost model
test = np.array([holder[1:]])

# load the model and predict the transformed input variables
model = joblib.load(open('house_model.pkl', 'rb'))
model_pred = model.predict(test)
proba_scores = model.predict_proba(test)

from datetime import date
today_date = date.today()

if st.button('PREDICT'):
    if model_pred == 1:
        st.success('ELITE HOUSING')
        st.text(f"Probability Score: {proba_scores}")
        st.image("image\pngwing.com (30).png", caption = 'This is your elite house type', width = 200)
        st.info(f"predicted at: {today_date}")

    elif model_pred == 2:
        st.success('LOWCOST HOUSING')
        st.text(f"Probability Score: {proba_scores}")
        st.image('image\pngwing.com (31).png', caption = 'This is your Lowcost Housing', width = 200)
        st.info(f"predicted at: {today_date}")

    else:
        st.success('AVERAGE')
        st.text(f"Probability Score: {proba_scores}")
        st.image('image\pngwing.com (27).png', caption = 'This is your average housing', width = 200)
        st.info(f"predicted at: {today_date}")


st.markdown("<br>", unsafe_allow_html = True )
# - Check for Multi-Colinearity in the chosen features.
heat = plt.figure(figsize = (14, 7))
sns.heatmap(new_feats.corr(), annot = True, cmap = 'BuPu')

st.write(heat)

st.markdown("<hr><hr>", unsafe_allow_html= True)


