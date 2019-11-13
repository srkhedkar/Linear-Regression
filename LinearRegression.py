#import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#to plot within notebook
import matplotlib.pyplot as plt

#read the data file
df = pd.read_csv('D:\\python3\\data\\SensexHistoricalData.csv')

#setting index as date
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']

#converting dates into number of days as dates cannot be passed directly to any regression model
df.index = (df.index - pd.to_datetime('1970-01-01')).days

# Convert the pandas series into numpy array, we need to further massage it before sending it to regression model
y = np.asarray(df['Close'])
x = np.asarray(df.index.values)

# Model initialization
# by default the degree of the equation is 1.
# Hence the mathematical model equation is y = mx + c, which is an equation of a line.
regression_model = LinearRegression()

# Fit the data(train the model)
regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

# Prediction for historical dates. Let's call it learned values.
y_learned = regression_model.predict(x.reshape(-1, 1))

# Now, add future dates to the date index and pass that index to the regression model for future prediction.
# As we have converted date index into a range index, hence, here we just need to add 3650 days ( roughly 10 yrs)
# to the previous index. x[-1] gives the last value of the series.
newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + 3650))

# Prediction for future dates. Let's call it predicted values.
y_predict = regression_model.predict(newindex.reshape(-1, 1))

#print the last predicted value
print ("Closing price at 2029 would be around ", y_predict[-1])

#convert the days index back to dates index for plotting the graph
x = pd.to_datetime(df.index, origin='1970-01-01', unit='D')
future_x = pd.to_datetime(newindex, origin='1970-01-01', unit='D')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#plot the actual data
plt.figure(figsize=(16,8))
plt.plot(x,df['Close'], label='Close Price History')

#plot the regression model
plt.plot(x,y_learned, color='r', label='Mathematical Model')

#plot the future predictions
plt.plot(future_x,y_predict, color='g', label='Future predictions')

plt.suptitle('Stock Market Predictions', fontsize=16)

fig = plt.gcf()
fig.canvas.set_window_title('Stock Market Predictions')

plt.legend()
plt.show()