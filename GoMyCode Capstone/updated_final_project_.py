
# %pip install requests --q 
# %pip install pmdarima --q
# from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = 'darkgrid')
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# %matplotlib inline
import requests


"""***GET DATA***"""

# %pip install meteostat --q
from meteostat import Stations, Daily
# Using the Meteostat API, we need the station ID of each state in Nigeria and the data start and ending dates

# Get the station IDs for each state in Nigeria
# Connect the API with the station ID and the start/end date

def get_data(state, start_date, end_date):
    station = Stations()
    nig_stations = station.region('NG') #...................................... Filter stations by country code (NG for Nigeria)
    nig_stations = nig_stations.fetch() # ...................................... Fetch the station information

    # Now fetch the state's weather information if the fetched dataframe contains the state name
    state_stations = nig_stations[nig_stations['name'].str.contains(state)]

    #............ Some state names have a '/' in them. So we clean them up
    nig_stations['name'] = nig_stations['name'].apply(lambda x: x.split('/', 1)[0])
    nig_stations.drop_duplicates(subset = ['name'], keep = 'first', inplace =True)
    
    if len(state_stations) > 0:
        station_id = nig_stations.index[0]
    else: None

    data = Daily(station_id, start_date, end_date)
    data = data.fetch()

    return data

# Get current date so we could use it as stop date
from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')

weather_data = get_data('Yola', '2021-01-01', str(current_date))
weather_data.head()

"""***GET COORDINATES***"""

# We can also use GEOPY library to get coordinates

def get_coordinates():
    from geopy.geocoders import Nominatim

    # Create a geocoder instance
    geolocator = Nominatim(user_agent="my-app")

    # List of Nigerian cities
    states = [
        'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
        'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'FCT - Abuja', 'Gombe',
        'Imo', 'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos',
        'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers', 'Sokoto',
        'Taraba', 'Yobe', 'Zamfara'
    ]

    # Retrieve latitude and longitude for each city
    coordinates = []
    for city in states:
        location = geolocator.geocode(city + ", Nigeria")
        if location is not None:
            coordinates.append([city, location.latitude, location.longitude])
    
    global nigerian_states
    # Print the list of cities with their corresponding latitude and longitude
    nigerian_states = []
    for city_data in coordinates:
        nigerian_states.append(city_data)

    global just_states
    # Using list comprehension, access the first items (states) in all the lists and save it to 'States'
    just_states = []
    for sublist in nigerian_states:
        first_item = sublist[0]
        just_states.append(first_item.lower())

    return nigerian_states, just_states

coordinates, states = get_coordinates()
print(f"{coordinates[:4]}\n\n")
print(states[:10])

"""Having Gotten the historical data needed, then we can predict the following.<br>

- Rainfall: We will use Temperature(mean), Wind-speed, and Pressure to predict the Precipitation (rainfall) using RandomForest Algorithm.
- Temperature, Pressure, Wind-speed: We will use Arima Time Series for prediction of these variables. <br><hr>
"""

# Next, we will create a dataframe for the precipitation (rainfall) prediction

# Solve For Average Temperature
def find_mean(frame, desired_name, concerned_columns):
    new_frame = pd.DataFrame()
    new_frame[desired_name] = frame[concerned_columns].mean(axis = 1)
    return new_frame
temp = find_mean(weather_data, 'avg_temp', ['tmin', 'tmax'])

press = weather_data.pres #..................................................... Solve for Pressure
wind_speed = weather_data.wspd # ............................................... Solve for Wind-Speed
precipe = weather_data.prcp

# Bring all data together into one dataframe
rain_df = pd.concat([temp, press, wind_speed, precipe], axis = 1)
rain_df.head()

"""***Classify Precipitation To Ascertain Rainfall.***
- Light rain — when the precipitation rate is < 2.5 mm (0.098 in) per hour. 
- Moderate rain — when the precipitation rate is between 2.5 mm (0.098 in) – 7.6 mm (0.30 in) or 10 mm (0.39 in) per hour. 
- Heavy rain — when the precipitation rate is > 7.6 mm (0.30 in) per hour, or between 10 mm (0.39 in) and 50 mm (2.0 in)
"""

# Modelling For Precipitaion
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


def modelling(data):

    # preprocess the data
    scaler = StandardScaler()
    x = rain_df.drop('prcp', axis = 1)
    for i in x.columns:
        x[[i]] = scaler.fit_transform(x[[i]])
    y = rain_df.prcp

    # find the best random-state
    model_score = []
    for i in range(1, 100):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = i)
        rf_model = RandomForestRegressor()
        rf_model.fit(xtrain, ytrain)
        prediction = rf_model.predict(xtest)
        score = r2_score(ytest, prediction)
        model_score.append(score)

    best_random_state = [max(model_score), model_score.index(max(model_score))]

    # Now we start training and since we have gotten best random_state
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = best_random_state[1] + 1)
    rf_model = RandomForestRegressor()
    rf_model.fit(xtrain, ytrain)
    prediction = rf_model.predict(xtest)
    score = r2_score(ytest, prediction)
    print(f"Coefficient of Determination: {score}")

    return rf_model, best_random_state
    
rain_model, best_score = modelling(rain_df)
print(best_score)

"""<hr>

***ARIMA TIME SERIES FOR TEMPERATURE, PRESSURE AND WIND-SPEED***
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install numpy scipy patsy statsmodels --q
from statsmodels.tsa.stattools import adfuller # ............ Import the adfuller library that runs the stationary test
from statsmodels.tsa.arima_model import ARIMA

# Create a function that tests for the stationarity of the dataset, and stationarises it if need be.

def stationarizer(dataframe, response_variable):
  result = adfuller(dataframe[response_variable], regression = 'ct') #.............................. Save the result of the adfuller test in a variable called RESULT

  ADF_stats = result[0] # ....................................................... create a container for the first adfuller test statistic result
  p_values = result[1] # ........................................................ create a container for the second adfuller test statistic result
  critical_values = [] # ........................................................ create a container for the selected critical value.
  for keys, values in result[4].items(): # ...................................... Select the preferred critical value and save it inside the container above
    critical_values.append(values)

  # Create a statement that prints if the dataset is stationary
  if ADF_stats < critical_values[1]:
    print('stationary')
    return dataframe[[response_variable]]
  else: 
    print('None Stationary, hence stationarized')
    return (dataframe[[response_variable]] - dataframe[[response_variable]].shift(1))

# Get time and response variable
pressure = rain_df[['pres']]
temp = rain_df[['avg_temp']]
wind = rain_df[['wspd']]

# Run the stationarizer on the dataframe
press = stationarizer(pressure, 'pres')
press.dropna(inplace = True)

temp = stationarizer(temp, 'avg_temp')
temp.dropna(inplace = True)

wind = stationarizer(wind, 'wspd')
wind.dropna(inplace = True)

def plotter(dataframe):
    sns.set
    plt.figure(figsize = (15, 3))
    plt.subplot(1,1,1)
    sns.lineplot(dataframe)

plotter(press)
plotter(temp)
plotter(wind)

# .............Determining lag value for our time series model by looping through possible numbers. This method is called GRID SEARCH

import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

def best_parameter(data):
    # Create a grid search of possible values of p,d,and q values
    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 4)
    # p = d = q = range(2,6)

    # Create a list to store the best AIC values and the corresponding p, d, and q values
    best_aic = np.inf
    best_pdq = None

    # Loop through all possible combinations of p, d, and q values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Fit the ARIMA model
                model = sm.tsa.arima.ARIMA(data, order=(p, d, q))
                try:
                    model_fit = model.fit()
                    # Update the best AIC value and the corresponding p, d, and q values if the current AIC value is lower
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_pdq = (p, d, q)
                except:
                    continue
    return best_pdq

pressure_param = best_parameter(pressure)
wind_param = best_parameter(wind)
temp_param = best_parameter(temp)
print(f"Best Parameter For Pressure: {best_parameter(pressure)}")
print(f"Best Parameter For Temperature: {best_parameter(temp)}")
print(f"Best Parameter For WindSpeed: {best_parameter(wind)}")

from statsmodels.tsa.arima.model import ARIMA
# ......................Create a function that models, predict, and returns the model and the predicted values...........

def arima_model(data, response_column, best_param, prediction_frequency, prediction_period):
    plt.figure(figsize = (14,3))
    model = ARIMA(data, order = best_param)
    model = model.fit()

    # Plot the actual data and the prediction
    plt.plot(data, color = 'blue')
    plt.plot(model.fittedvalues, color = 'red')
    plt.title('RSS: %.4F'% sum((model.fittedvalues-data[response_column])**2))
    print('plotting AR model')

    # Plot the Future Predictions according to the model
    future_dates = pd.date_range(start = data.index[-1], periods = prediction_period, freq = prediction_frequency)
    forecast = model.predict(start = len(data), end = len(data) + (prediction_period - 1))
    plt.figure(figsize = (14,4))
    plt.plot(data, label='Original Data', color = 'blue')
    plt.plot(future_dates, forecast, color='red', label='Predictions')
    plt.plot(model.fittedvalues, color = 'red')
    plt.title('ARIMA Model: Future Predictions')
    plt.xlabel('Date')
    plt.ylabel(response_column)
    plt.legend()
    plt.show()

    # get the dataframes of the original/prediction
    data_ = data.copy()
    data_['predicted'] = model.fittedvalues

    # Get the dataframe for predicted values
    predictions_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    predictions_df.set_index('Date', drop = True, inplace = True)

    return model, data_, predictions_df

"""Explanation of the Modelling Function:
- data: The dataframe we want to model
- Response_column: The particular column in the dataframe
- best_param: The best parameter giving in the previous function.
- prediction_frequency: The frequency we want our prediction to come (daily, weekly, monthly or yearly as 'D', 'W', 'M', 'Y' respectively)
- prediction_period: If prediction_frequency is Daily, then prediction_period is the number of days we want to predict. <hr>

Now we plot the model and their prediction for the next 5 days.<br>
It is important to note that the model weakens as the forecasting horizon extends further into the future. Hence we limit prediction to maximum of 10 days
"""

pressure_model, pressure_dataframe, pressure_predicted_df = arima_model(press, 'pres', pressure_param, 'D', 5)

temp_model, temp_dataframe, temp_predicted_df = arima_model(temp, 'avg_temp', temp_param, 'D', 5)

wind_model, wind_dataframe, wind_predicted_df = arima_model(wind, 'wspd', wind_param, 'D', 5)





word = 'I want to find out if it will rain in Zamfara'
place = [i for i in word.lower().split() if i in just_states][0] # ............. Extracts out name of state mentioned in the input statement
cord = [sublist for sublist in coordinates if place in sublist[0].lower()][0] #. Fetches the coordinate of the state name extracted above
latitude, longitude  = cord[1], cord[2] # ...................................... Saves it to variable name longitude and latitude
place = place.capitalize() 
print(f"The Longitude of {place} is {longitude}, and the Latitude is {latitude}")

word = 'What is the weather condition in lagos state'
place = [i.capitalize() for i in word.lower().split() if i in just_states][0]
place

get_data(state, start_date, end_date)

# Import Meteostat library and dependencies
# %pip install meteostat --q

plt.figure(figsize = (14,4))
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2017, 1, 1)
end = datetime(2023, 4, 30)

from meteostat import Stations

stations = Stations()
stations = stations.region('NG')  # Filter stations by country code (NG for Nigeria)
stations = stations.fetch()

lagos_stations = stations[stations['name'].str.contains('Lagos')]
# Create Point for Vancouver, BC
point = Point( {lagos_stations.iloc[:,5].values[0]}, {lagos_stations.iloc[:,6].values[0]})

# Get daily data from 2017 to 2023
data = Daily(point, start, end)
data = data.fetch()
data
# # Plot line chart including average, minimum and maximum temperature
# data.plot(y=['tavg', 'tmin', 'tmax'])
# plt.show()

from meteostat import Stations

stations = Stations()
nig_stations = stations.region('NG')  # Filter stations by country code (NG for Nigeria)
us_stations = stations.region('US')
nig_stations = nig_stations.fetch()
us_stations = us_stations.fetch()
# lagos_stations = stations[stations['name'].str.contains('Lagos')]
# lagos_stations.index[0]
# nig_stations['name'] = (lambda i:i.split('/') for i in nig_stations['name'])
nig_stations['name'] = nig_stations['name'].apply(lambda x: x.split('/', 1)[0])
nig_stations.drop_duplicates(subset = ['name'], keep = 'first', inplace =True)