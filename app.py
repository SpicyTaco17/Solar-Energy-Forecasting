import streamlit as st
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.title('Solar Energy Forecasting')
st.image('https://th.bing.com/th/id/R.d18dc3d25500a1e180088b0d348b3a05?rik=axhPjpc0Xzp41Q&pid=ImgRaw&r=0')

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

xgbmodel = xgb.XGBRegressor()
xgbmodel.fit(xTrain, yTrain)
predictions = xgbmodel.predict(xTest)
predictions_test = xgbmodel.predict(xTTest)

panels = st.number_input("Number of panels", min_value = 0, value = 0)
time_utc = st.time_input("Time (UTC)", step = 3600, value = datetime.time(12, 00))
temp = st.slider("Temperature (°C)", min_value = -20, max_value = 50, value = 15)
radiance = st.slider("Total radiance (W/m2)", min_value = 0, max_value = 1000, value = 500)

numeric_features = ['timestamp', 'FR_temperature', 'df_global_radiation']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)])

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regression', xgbmodel)
])

pipeline.fit(xTrain, yTrain)

if time_utc == datetime.time(0, 00):
    time_utc = 0
elif time_utc == datetime.time(1, 00):
    time_utc = 1
elif time_utc == datetime.time(2, 00):
    time_utc = 2
elif time_utc == datetime.time(3, 00):
    time_utc = 3
elif time_utc == datetime.time(4, 00):
    time_utc = 4
elif time_utc == datetime.time(5, 00):
    time_utc = 5
elif time_utc == datetime.time(6, 00):
    time_utc = 6
elif time_utc == datetime.time(7, 00):
    time_utc = 7
elif time_utc == datetime.time(8, 00):
    time_utc = 8
elif time_utc == datetime.time(9, 00):
    time_utc = 9
elif time_utc == datetime.time(10, 00):
    time_utc = 10
elif time_utc == datetime.time(11, 00):
    time_utc = 11
elif time_utc == datetime.time(12, 00):
    time_utc = 12
elif time_utc == datetime.time(13, 00):
    time_utc = 13
elif time_utc == datetime.time(14, 00):
    time_utc = 14
elif time_utc == datetime.time(15, 00):
    time_utc = 15
elif time_utc == datetime.time(16, 00):
    time_utc = 16
elif time_utc == datetime.time(17, 00):
    time_utc = 17
elif time_utc == datetime.time(18, 00):
    time_utc = 18
elif time_utc == datetime.time(19, 00):
    time_utc = 19
elif time_utc == datetime.time(20, 00):
    time_utc = 20
elif time_utc == datetime.time(21, 00):
    time_utc = 21
elif time_utc == datetime.time(22, 00):
    time_utc = 22
elif time_utc == datetime.time(23, 00):
    time_utc = 23
elif time_utc == datetime.time(24, 00):
    time_utc = 24

click = st.button('Get Prediction')

def results():
    user_input = pd.DataFrame()
    user_input.insert(0, "timestamp", [time_utc])
    user_input.insert(1, "FR_temperature", [temp])
    user_input.insert(2, "df_global_radiation", [radiance])
    prediction = pipeline.predict(user_input)
    final = prediction
    if final < 0:
        final = 0
    else:
        final = panels*(prediction / 15856860)
    st.text(final)

if click:
    st.subheader('Results (MW):')
    st.balloons()
    results()
