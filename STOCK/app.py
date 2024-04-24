import numpy as np 
import pandas as pd 
import yfinance as yf 
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import joblib

model = load_model('C:/Users/user/PYTHON_PROJECT/STOCK/Stock Predications Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symnbol', 'GC=F')
start = st.text_input('Enter Start Date', '2012-01-01')
end = st.text_input('Enter End Date','2023-12-31')

data = yf.download(stock,start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2) 

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3) 



x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label= 'Predicted Price')
plt.plot(y, 'g', label ='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4) 

st.subheader('Future Price Prediction')
data2 = yf.Ticker(stock)
data2 = data2.history(period="max")
del data2["Dividends"]
del data2["Stock Splits"]
data2["Tomorrow"] = data2["Close"].shift(-1)
data2["Target"] = (data2["Tomorrow"] > data2["Close"]).astype(int)
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = data2.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    data2[ratio_column] = data2["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    data2[trend_column] = data2.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]
data2 = data2.dropna()
model2 = joblib.load('Stock Predications Regression Model.joblib')
train = data2.iloc[:-100]
test = data2.iloc[-100:]

predictors = ['Close_Ratio_2', 'Trend_2', 'Close_Ratio_5', 'Trend_5',
       'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250',
       'Close_Ratio_1000', 'Trend_1000']

predict2 = model2.predict_proba(test[predictors])[:,1]
predict2[predict2 >= .6] = 1
predict2[predict2 < .6] = 0
fig5 = plt.figure(figsize=(8,6))
plt.plot(predict2, 'r', label='Future Predicted Price')
plt.plot(y, 'g', label =  'Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig5)