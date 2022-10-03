import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
st.title("PREDICT")
start = "2015-01-01"
today = datetime.today().strftime("%Y-%m-%d")
stocks = ("AAPL",'GOOG','MSFT','GME')
selected = st.selectbox('company',stocks)
n_years = st.slider("Years of predcition",1,4)
period = n_years*365
@st.cache
def load_data(ticker):
    data = yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected)
data_load_state.text("Done...")
st.subheader('RAW DATA')
st.write(data .tail())
def plot_raw():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y= data['Open'], name="OPEN"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data['Close'], name="CLOSE"))
    fig.layout.update(title_text="TIME_SERIES_DATA",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)
st.subheader('Forecast data')
st.write(forecast.tail())

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)