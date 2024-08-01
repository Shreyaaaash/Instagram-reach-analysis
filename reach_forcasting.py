
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
import warnings
pio.templates.default = "plotly_white"

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.day_name()
    return data

def analyze_daily_stats(data):
    day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['mean'], name='Mean'))
    fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['median'], name='Median'))
    fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['std'], name='Standard Deviation'))
    fig.update_layout(title='Instagram Reach by Day of the Week', xaxis_title='Day', yaxis_title='Instagram Reach')
    fig.show()

def fit_sarima_model(data, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                      order=order,
                                      seasonal_order=seasonal_order)
    return model.fit()

def forecast_and_visualize(data, model, forecast_periods):
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods+1)
    predictions = model.predict(start=len(data), end=len(data)+forecast_periods)
    
    trace_train = go.Scatter(x=data['Date'], y=data["Instagram reach"], mode="lines", name="Training Data")
    trace_pred = go.Scatter(x=future_dates, y=predictions, mode="lines", name="Predictions")
    
    layout = go.Layout(title="Instagram Reach Time Series and Predictions", 
                       xaxis_title="Date", 
                       yaxis_title="Instagram Reach")
    
    fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
    fig.show()

def main():
    data = load_data("data/Instagram-Reach.csv")
    analyze_daily_stats(data)
    model = fit_sarima_model(data, order=(8,1,2), seasonal_order=(8,1,2,12))
    print(model.summary())
    forecast_and_visualize(data, model, 100)

if __name__ == "__main__":
    main()