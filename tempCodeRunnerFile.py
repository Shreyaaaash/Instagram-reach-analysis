import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
import warnings
pio.templates.default = "plotly_white"

data = pd.read_csv("data/Instagram-Reach.csv", encoding='latin-1')
data['Date'] = pd.to_datetime(data['Date'])
print(data.head()) #to load the data

data['Day'] = data['Date'].dt.day_name()

day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)#to group by day

fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['mean'], name='Mean'))
fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['median'], name='Median'))
fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['std'], name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', xaxis_title='Day', yaxis_title='Instagram Reach')
fig.show()#to assign x and y cordinate to the 

p, d, q = 8, 1, 2
import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())