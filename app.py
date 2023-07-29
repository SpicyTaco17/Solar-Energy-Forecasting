!pip3 install xgboost
import streamlit as st
import pandas as pd
import numpy as np

st.title('Solar Energy Forecasting')

df_1 = pd.read_csv('https://raw.githubusercontent.com/SpicyTaco17/Solar-Energy-Forecasting/main/time_series_60min_singleindex_filtered.csv')
df_2 = pd.read_csv('https://raw.githubusercontent.com/SpicyTaco17/Solar-Energy-Forecasting/main/weather_data_filtered.csv')
df_3 = pd.read_csv('https://raw.githubusercontent.com/SpicyTaco17/Solar-Energy-Forecasting/main/timestamp_utc.csv')

df_total = df_3.join(df_2.iloc[:,-3:]).join(df_1.iloc[:,-3:]).iloc[:,1:]
df_global_radiation = df_total['FR_radiation_diffuse_horizontal'] + df_total['FR_radiation_direct_horizontal']

df_total = df_total.drop(columns = ['FR_radiation_direct_horizontal'])
df_total = df_total.drop(columns = ['FR_radiation_diffuse_horizontal'])
df_total = df_total.drop(columns = ['FR_load_actual_entsoe_transparency'])
df_total = df_total.drop(columns = ['FR_load_forecast_entsoe_transparency'])

global_radiation = pd.Series(df_global_radiation)
df_total = df_total.assign(df_global_radiation = global_radiation)

df_total = df_total.dropna(how='any',axis=0)

ground_truth = df_total['FR_solar_generation_actual'].copy()
parameter_dataset = df_total.drop('FR_solar_generation_actual', axis=1)

(xTr, xTTest, yTr, yTTest) = train_test_split(parameter_dataset, ground_truth, test_size = .2, shuffle=False)
(xTrain, xTest, yTrain, yTest) = train_test_split(xTr, yTr, test_size = .4, random_state=17)

# model = pd.read_csv()

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache_data)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)
