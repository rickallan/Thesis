import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import kurtosis
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.svm import SVR # for building the model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#import data
BC=pd.read_csv(r'')
Cap=pd.read_csv(r'')
Temp=pd.read_csv(r'')
TTF=pd.read_csv(r'')
TTF = TTF.rename(columns={'Close Price': 'ClosePrice'})

#convert Capacity from % to decimals
Cap['decimal'] = Cap['Capacity'].apply(lambda x: float(x.strip('%')) / 100)
Cap = Cap.drop('Capacity',axis=1)
Cap = Cap.rename(columns={'decimal': 'Capacity'})

BC = BC.rename(columns={' Coal_Price': 'Coal_Price'})
#convert dfs to time series by setting date as an index
TTFts=TTF.set_index("Date")
Capts=Cap.set_index("Date")
Tempts=Temp.set_index("Date")
BCts=BC.set_index("Date")

#convert all indexes to datetime objects(all indexes need to be in the same format in order to merge)

TTFts.index = pd.to_datetime(TTFts.index,dayfirst=True)
Tempts.index = pd.to_datetime(Tempts.index)
Capts.index = pd.to_datetime(Capts.index)
BCts.index = pd.to_datetime(BCts.index)

BCts=BCts.resample('D').mean()
BCts = BCts.ffill()

#replace -999 temperature data entry error with the average
avg_temperature=Tempts['Temperature'].mean()
for i in range(len(Tempts)):
               if Tempts.iloc[i,0]<-900:
                   Tempts.iloc[i,0]=avg_temperature
#merge dfs
df = pd.merge(TTFts, Tempts, left_index=True,right_index=True).merge(Capts, left_index=True, right_index=True).merge(BCts, left_index=True,right_index=True)
#drop NAs
df=df.dropna()

# TTF Plot
plt.plot(df.index, df.ClosePrice)

# add titles and labels
plt.title('TTF Time Series')
plt.xlabel('Date')
plt.ylabel('TTF Close Price')

# display the plot
#plt.show()
#clear plots
#plt.clf()


# Temperature Plot
plt.plot(df.index, df.Temperature)

# add titles and labels
plt.title('Temperature Time Series')
plt.xlabel('Date')
plt.ylabel('Avg of Western Europe Temperature')


#Capacity plot
plt.plot(df.index, df.Capacity)


# add titles and labels
plt.title('Capacity Time Series')
plt.xlabel('Date')
plt.ylabel('Capacity')

# display the plot
#plt.show()

#Coal Price plot
plt.plot(df.index, df.Coal_Price)


# add titles and labels
plt.title('Coal Price Time Series')
plt.xlabel('Date')
plt.ylabel('Coal Price')

# display the plot
#plt.show()

#Coal Price plot
plt.plot(df.index, df.Brent_Price)


# add titles and labels
plt.title('Brent Price Time Series')
plt.xlabel('Date')
plt.ylabel('BrentPrice')

# display the plot
#plt.show()

#descriptive statistics
stats=df.describe()
print(stats)

kurt = kurtosis(df)
# a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution
print("Kurtosis:", kurt)

# a measurement of the distortion of symmetrical distribution or asymmetry in a data set.
print("Skewness:", df.skew())

#df correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

#PACF and ACF Plots for autocorrelation
plot_acf(df['ClosePrice'], lags=20)
#plt.show()
plot_pacf(df['ClosePrice'], lags=20)
#plt.show()

cpfrequency='D'
decompose_result = seasonal_decompose(df['ClosePrice'], model='additive', period=7)
#plt.show()

#decompose_result1 = seasonal_decompose(df['ClosePrice'],model='additive')
#decompose_result1.plot()
#decompose_result2 = seasonal_decompose(df['ClosePrice'],model='multiplicative')
#decompose_result2.plot()



#####MODELS####

#DATASET SPLIT
df=df.reset_index()
#export df to excel
#df.to_excel('df.xlsx', index=False)

#split dataset into dependent variable and regressors
Dependent=df.loc[:,['Date','ClosePrice']]
Regressors=df.loc[:,['Date','Temperature','Capacity','Brent_Price','Coal_Price']]

#80%
Dependent_train=Dependent.head(1838)
Regressors_train=Regressors.head(1838)

#20%
Dependent_test=Dependent.tail(459)
Regressors_test=Regressors.tail(459)


#SVR

from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.svm import SVR # for building the model
from sklearn.model_selection import train_test_split

x = Regressors
y = Dependent


#set index in order to proceed to scaling
x=x.set_index('Date')
y=y.set_index('Date')


sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#split for arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#GRID SEARCH, SVR OPTIMIZER, required to run once
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear', 'rbf']}
svr=SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(x_train, y_train.ravel())
print(grid_search.best_params_)

model = SVR(kernel='rbf', C=10, gamma=1)
model.fit(x_train, y_train.ravel())


y_pred = model.predict(x_test)


#random results, run 10 times and report the average of the performance metrics
print("SVR Results")
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)
mape=mean_absolute_percentage_error(y_test,y_pred)
print("MAPE: ", mape)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
#SVR with diff, try stationarity to see if better results are yielded

df1=df
df1=df1.set_index('Date')
df1_diff = df1.diff().dropna()
df1_diff=df1_diff.reset_index()

y=df1_diff.loc[:,['Date','ClosePrice']]
x=df1_diff.loc[:,['Date','Temperature','Capacity','Brent_Price','Coal_Price']]

y=y.set_index('Date')
x=x.set_index('Date')

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.model_selection import GridSearchCV
param_grid2 = {'C': [0.1, 1, 10],
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear', 'rbf']}
svr=SVR()
grid_search = GridSearchCV(svr, param_grid2, cv=5)
grid_search.fit(x_train, y_train.ravel())
print(grid_search.best_params_)


model = SVR(kernel='rbf', C=0.1, gamma=0.1)
model.fit(x_train, y_train.ravel())

y_pred = model.predict(x_test)
print("SVR 1st Diff Results")
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)
mape=mean_absolute_percentage_error(y_test,y_pred)
print("MAPE: ", mape)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

#SARIMAX

#preparation

#ADF Test
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df.ClosePrice, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

#p=value =0,22 , non stationary time series, requires differencing


import statsmodels.api as sm

# Set the date column as the index
Dependent_train=Dependent_train.set_index("Date")
Regressors_train=Regressors_train.set_index("Date")
Dependent_test=Dependent_test.set_index("Date")
Regressors_test=Regressors_test.set_index("Date")


# Create the SARIMAX model with 5 regressors
#model = sm.tsa.SARIMAX(Dependent_train, exog=Regressors_train, order=(2,1,2), seasonal_order=(0,0,0,0))

from pmdarima import auto_arima
model = auto_arima(Dependent_train, exogenous=Regressors_train, seasonal=True, max_order=None, suppress_warnings=True, stepwise=True)
print(model.summary())
sarimaxforecasts=model.predict(n_periods=len(Dependent_test), exogenous=Regressors_test)
#sarimaxforecasts = results.get_forecast(steps=326, exog=Regressors_test)


# Fit the model
#results = model.fit()
#sarimaxforecasts=results.predict(Regressors_test)
#sarimaxforecasts = results.get_forecast(steps=326, exog=Regressors_test)


print("SARIMAX Results")
rmse = np.sqrt(mean_squared_error(Dependent_test, sarimaxforecasts))
mape=mean_absolute_percentage_error(Dependent_test, sarimaxforecasts)
mae = mean_absolute_error(Dependent_test, sarimaxforecasts)
r2 = r2_score(Dependent_test, sarimaxforecasts)

#mape = mean_absolute_percentage_error(Dependent_test, sarimaxforecasts.predicted_mean)
#rmse = np.sqrt(mean_squared_error(Dependent_test, sarimaxforecasts.predicted_mean))

print("\nRMSE: ", rmse)
print("MAPE: ", mape)
print("MAE: ", mae)
print("R-squared:", r2)

#Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
LRX_train=Regressors_train
LRY_train=Dependent_train
#LRX_train = LRX_train.reset_index(drop=True)
#LRY_train = LRY_train.reset_index(drop=True)

LRX_test=Regressors_test
LRY_test=Dependent_test
#LRX_test = LRX_test.reset_index(drop=True)
#LRY_test = LRY_test.reset_index(drop=True)


#X_train, X_test, y_train, y_test = train_test_split(LRX, LRY, test_size=0.2)
lr = LinearRegression().fit(LRX_train, LRY_train)
lry_pred = lr.predict(LRX_test)

print("Linear Regression Results")
mape = mean_absolute_percentage_error(LRY_test, lry_pred)
rmse = np.sqrt(mean_squared_error(LRY_test, lry_pred))
mae = mean_absolute_error(LRY_test, lry_pred)
r2 = r2_score(LRY_test, lry_pred)

print("\nRMSE: ", rmse)
print("\nMAPE: ", mape)
print("MAE: ", mae)
print("R-squared:", r2)


#RNN
# load the dataset
dfrnn = df
dfrnn=dfrnn.set_index('Date')


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


train_data = dfrnn.head(1838)
test_data = dfrnn.tail(459)

# Select target variable and regressors
target_var = 'ClosePrice'
regressors = ['Temperature', 'Capacity', 'Coal_Price', 'Brent_Price']

# Split data into train and test sets
train_data = df.head(1838)
test_data = df.tail(459)

# Scale data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[[target_var] + regressors])
test_data_scaled = scaler.transform(test_data[[target_var] + regressors])

# Define function to create samples
def create_samples(data, n_steps):
    X, Y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps, 1:])
        Y.append(data[i+n_steps, 0])
    return np.array(X), np.array(Y)

# Set number of time steps
n_steps = 5

# Create train and test samples
X_train, Y_train = create_samples(train_data_scaled, n_steps)
X_test, Y_test = create_samples(test_data_scaled, n_steps)

# Define LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps, len(regressors))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X_train, Y_train, epochs=200, verbose=0)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, test_data_scaled[-len(y_pred):, 1:]), axis=1))[:, 0]
Y_test_inv = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1,1), test_data_scaled[-len(y_pred):, 1:]), axis=1))[:, 0]

print('RNN-LSTM Results')
rmse = np.sqrt(mean_squared_error(Y_test_inv, y_pred_inv))
mape=mean_absolute_percentage_error(Y_test_inv,y_pred_inv)
mae = mean_absolute_error(Y_test_inv,y_pred_inv)
r2 = r2_score(Y_test_inv,y_pred_inv)

print("\nRMSE: ", rmse)
print("\nMAPE: ", mape)
print("MAE: ", mae)
print("R-squared:", r2)
