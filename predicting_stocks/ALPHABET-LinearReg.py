import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import math
import datetime
import seaborn

from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

style.use('ggplot')

original_data = quandl.get('WIKI/GOOGL')
df = original_data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

label = 'Adj. Close'
#replace all NaNs as outliers
df.fillna(-9999,inplace = True)

forecast_out = int(math.ceil(.01*len(df)))

df['label'] = df[label].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .1)
clf = LinearRegression(n_jobs = 5)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

confidence = clf.score(X_forecast[:-1], X_forecast[-1])

#predict the next .1(data length) days ~ 33
forcast_set = clf.predict(X_forecast)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.plot(y[-33:])
plt.legend(loc= 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show(block = False)
