"""This is the main script to run.
Time Series for TIA.
"""
import os
import sys
import time
import logging
import datetime as dt
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, kstest, chisquare, \
    jarque_bera
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, \
    SimpleExpSmoothing, Holt
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import hurst
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from fbprophet import Prophet

start_time = time.time()
np.random.seed(7)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
current_Date = dt.datetime.today().strftime('%d-%b-%Y-%H-%M-%S')
log_filename = 'std-' + current_Date + '.log'
logging.basicConfig(filename=ROOT_DIR + '/logs/' + log_filename,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    encoding='utf-8')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("STARTING to run the project!")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 10)
pd.set_option("max_colwidth", 1000)
matplotlib.use('Qt5Agg')
plt.rcParams.update({'figure.max_open_warning': 0})
START_DATE: str = '1946-01'

#######################################################################

# Manipulacion de archivos para lectura de informacion
SERIE_FILENAME: str = 'serie.dat'
SERIE_FILEPATH: str = '\\'.join([ROOT_DIR, 'data', SERIE_FILENAME])

#######################################################################

# Carga de dataset
try:
    my_series = pd.read_csv(SERIE_FILEPATH, header=None,
                            dtype=float).squeeze("columns")
except Exception as e:
    print(e)
    print("El dataset contiene mas de una columna")
    sys.exit()

period: int = len(my_series)
if period < 60:
    print("El dataset tiene menos de 60 meses")
    sys.exit()

series_index = pd.date_range(
    start=START_DATE, periods=period, freq='M').strftime('%Y-%m')
my_series.index = pd.to_datetime(series_index)
my_series.index = my_series.index.strftime('%Y-%m')
my_series = my_series.rename(None)
my_series.index = pd.to_datetime(my_series.index, format='%Y-%m')
print(my_series)

#######################################################################

train, test = train_test_split(my_series, test_size=0.2, shuffle=False)

#######################################################################

my_series.plot()
plt.ylabel('Dependant variable')
plt.title('Dependant variable records since January of 1946')
plt.show()

fig = plt.figure(2)
groups = my_series.groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
plt.show()

# Check for Time Series to follow Normal Distribution
normal_distribution: bool

# Shapiro-Wilk
normal_sw_bool: bool
stat_sw, p_sw = shapiro(my_series)
print('Shapiro-Wilk: %.3f, p=%.3f' % (stat_sw, p_sw))
alpha_sw: float = 0.05
if p_sw > alpha_sw:
    print('Sample looks Gaussian (fail to reject H0)')
    normal_sw_bool = True
else:
    print('Sample does not look Gaussian (reject H0)')
    normal_sw_bool = False

# D’Agostino’s K-squared
normal_k2_bool: bool
stat_n, p_n = normaltest(my_series)
print('D’Agostino’s K-squared: %.3f, p=%.3f' % (stat_n, p_n))
alpha_n = 0.05
if p_n > alpha_n:
    print('Sample looks Gaussian (fail to reject H0)')
    normal_k2_bool = True
else:
    print('Sample does not look Gaussian (reject H0)')
    normal_k2_bool = False

# Anderson-Darling
normal_ad_bool: bool
result = anderson(my_series)
print('Anderson-Darling: %.3f' % result.statistic)
p = 0
anderson_darling_list: list[bool] = []
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        normal_ad_bool = True
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        normal_ad_bool = False
    anderson_darling_list.append(normal_ad_bool)

# Kolmogorov Smirnov
normal_ks_bool: bool
ks_statistic, p_value = kstest(my_series, 'norm')
print("Kolmogorov Smirnov:", ks_statistic, p_value)
if p_value < 0.05:
    normal_ks_bool = False
else:
    normal_ks_bool = True

# Lilliefors
normal_ks_bool: bool
ksstat, pvalue = lilliefors(my_series)
print("Lilliefors:", ksstat, pvalue)
if pvalue < 0.05:
    normal_ks_bool = False
else:
    normal_ks_bool = True

# Chi-Square
normal_ch_bool: bool
statistic, pvalue_ch = chisquare(my_series)
print("Chi-Square:", statistic, pvalue_ch)
if pvalue_ch < 0.05:
    normal_ch_bool = False
else:
    normal_ch_bool = True

if period > 2000:
    # Jarque-Bera
    normal_jb_bool: bool
    statistic_jb, pvalue_jb = jarque_bera(my_series)
    print("Jarque-Bera:", statistic_jb, pvalue_jb)
    if pvalue_jb < 0.05:
        normal_jb_bool = False
    else:
        normal_jb_bool = True

# check for all tests to have at least one rejected hypotesis
print(anderson_darling_list, normal_k2_bool, normal_sw_bool, normal_ks_bool,
      normal_ks_bool, normal_ch_bool)
if (not any(anderson_darling_list)) or not normal_k2_bool or \
        not normal_sw_bool or not normal_ks_bool or not normal_ks_bool or \
        not normal_ch_bool:
    print("Serie no sigue distribucion normal")
    normal_distribution = False
else:
    normal_distribution = True

# Transformation Box-Cox
if not normal_distribution:
    print("A transformar con Box-Cox")
    fitted_data, fitted_lambda = stats.boxcox(my_series)
    fig, ax = plt.subplots(1, 2)
    sns.distplot(my_series, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 2},
                 label="Non-Normal", color="green", ax=ax[0])
    sns.distplot(fitted_data, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 2},
                 label="Normal", color="green", ax=ax[1])
    plt.legend(loc="upper right")
    fig.set_figheight(5)
    fig.set_figwidth(10)
    print(f"Lambda value used for Transformation: {fitted_lambda}")
    fitted_series = pd.Series(fitted_data)
    fitted_series.index = pd.to_datetime(series_index)
    fitted_series.index = fitted_series.index.strftime('%Y-%m')
    fitted_series.index = pd.to_datetime(fitted_series.index, format='%Y-%m')
    print(fitted_series)

    fig = plt.figure(4)
    fitted_series.plot()
    plt.ylabel('Fitted dependant variable')
    plt.title('Fitted dependant variable records since January of 1946')
    plt.show()
    my_series = fitted_series

# ESTACIONARIEDAD
# Augmented Dickey-Fuller
result_adf = adfuller(my_series, autolag='AIC')
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')
for key, value in result_adf[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    if result_adf[1] > 0.05 and result_adf[0] > value:
        print("No estacionaria")
    else:
        print("estacionaria")

# ESTACIONALIDAD
# Darts
series = TimeSeries.from_series(my_series)
is_seasonal, periodicity = check_seasonality(series, max_lag=240)
dict_seas = {
    "is seasonal?": is_seasonal,
    "periodicity (months)": f'{periodicity:.1f}',
    "periodicity (~years)": f'{periodicity / 12:.1f}'}
_ = [print(k, ":", v) for k, v in dict_seas.items()]

# Hurst
H, c, data_hurst = hurst.compute_Hc(my_series)
print("H = {:.4f}, c = {:.4f}".format(H, c))

# MODELOS
# Naive
train_df = train.to_frame(name='value')
test_df = test.to_frame(name='value')
train_len: int = len(train_df)
test_len: int = len(test_df)

y_hat_naive = test_df.copy()
y_hat_naive['naive_forecast'] = train_df['value'][train_len - 1]

rmse_naive = np.sqrt(mean_squared_error(
    test_df['value'], y_hat_naive['naive_forecast'])).round(2)

# Auto Regression
model = AutoReg(train_df, lags=22).fit()
forecasts = model.forecast(test_len).tolist()
forecast_df = pd.DataFrame(forecasts)
rmse_ar = mean_squared_error(test_df, forecast_df, squared=False)

# Seasonal Naive
df = my_series.to_frame('v')
decomposition = seasonal_decompose(x=df)
plt.rcParams.update({'figure.figsize': (16, 12)})
decomposition.plot().suptitle('Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Holt-Winters
model = Holt(np.asarray(train_df['value']))
model_fit = model.fit()
alpha_value = np.round(model_fit.params['smoothing_level'], 4)
Pred_Holt = test_df.copy()
Pred_Holt['Opt'] = model_fit.forecast(test_len)
rmse_opt = np.sqrt(mean_squared_error(test_df['value'], Pred_Holt['Opt']))

# ARIMA
model = ARIMA(train_df.values, order=(5, 0, 2))
model_fit = model.fit()
predictions = model_fit.predict(test_len)
print(predictions)
rmse_arima = sqrt(metrics.mean_squared_error(test_df.values,
                                             predictions[0:test_len]))

# Simple Exponential Smoothing
model = SimpleExpSmoothing(train)
alpha_list = [0.1, 0.5, 0.99]
pred_SES = test_df.copy()
for alpha_value in alpha_list:
    alpha_str = "SES" + str(alpha_value)
    mode_fit_i = model.fit(smoothing_level=alpha_value, optimized=False)
    pred_SES[alpha_str] = mode_fit_i.forecast(test_len)
    rmse_ses = np.sqrt(mean_squared_error(test_df, pred_SES[alpha_str]))
    print("RMSE is %3.4f" % rmse_ses)

# Triple Exponential Smoothing
model = ExponentialSmoothing(train_df.values)
model_fit = model.fit()
predictions_ = model_fit.predict(test_len)
print(predictions_)
rmse_tes = sqrt(metrics.mean_squared_error(test_df.values,
                                           predictions_[0:test_len]))

# LSTM
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(df)


def create_dataset(data, l_b=1):
    """This function convert an array of values into a daset matrix."""
    dataX, dataY = [], []
    for j in range(len(data) - l_b - 1):
        a = data[j:(j + l_b), 0]
        dataX.append(a)
        dataY.append(data[j + l_b, 0])
    return np.array(dataX), np.array(dataY)


# split into train and test sets
train, test = dataset[0:train_len, :], dataset[train_len:len(dataset), :]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
testScore = sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f'Test Score: {testScore} RMSE')

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] \
    = testPredict
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# # Prophet
# df.columns = ['ds', 'y']
# df['ds'] = pd.to_datetime(df['ds'])
# model = Prophet()
# model.fit(df)
# future = list()
# for i in range(1, test_len+1):
#     date = '1960-%02d' % i
#     future.append([date])
# future = pd.DataFrame(future)
# future.columns = ['ds']
# future['ds'] = pd.to_datetime(future['ds'])
# forecast = model.predict(future)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# model.plot(forecast)
# plt.show()

results = pd.DataFrame([{'Method': 'Naive method', 'RMSE': rmse_naive},
                        {'Method': 'Auto Regression', 'RMSE': rmse_ar},
                        {'Method': 'ARIMA', 'RMSE': rmse_arima},
                        {'Method': 'Triple Exponential Smoothing',
                         'RMSE': rmse_tes},
                        {'Method': 'Simple Exponential Smoothing',
                         'RMSE': rmse_ses},
                        {'Method': 'Holt-Winters', 'RMSE': rmse_opt},
                        {'Method': 'LSTM', 'RMSE': testScore}])
results.sort_values(by=['RMSE'], inplace=True)
results.to_csv('resultados.csv', index=False)
print(results)

#######################################################################

print("--- %s seconds ---" % (time.time() - start_time))
logger.info(f"--- {(time.time() - start_time)} seconds ---")
