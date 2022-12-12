import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams
from tensorflow.python.keras.optimizer_v2.adam import Adam

rcParams['figure.figsize'] = 16, 8

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

# pip install statsmodels
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess, Fourier

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

path = './input/store-sales-time-series-forecasting/'

##### 1. Events
df_hev = pd.read_csv(path + 'holidays_events.csv', parse_dates=['date'], infer_datetime_format=True)
df_hev['date'] = df_hev['date'].replace({'2013-04-29':
                                         pd.to_datetime('2013-03-29')}) # 'Good Friday' mistake correction
df_hev = df_hev.set_index('date').sort_index()
# df_hev = df_hev[df_hev.locale == 'National'] # National level only for simplicity
df_hev = df_hev.groupby(df_hev.index).first() # Keep one event only

print('df_hev')
print(df_hev)

##### 2. 주말일 경우 wd = false, 평일일 경우 wd = true
calendar = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31'))

calendar['dofw'] = calendar.index.dayofweek

calendar['wd'] = True
calendar.loc[calendar.dofw > 4, 'wd'] = False

calendar = calendar.merge(df_hev, how='left', left_index=True, right_index=True)

calendar.loc[calendar.type == 'Bridge', 'wd'] = False
calendar.loc[calendar.type == 'Work Day', 'wd'] = True
calendar.loc[calendar.type == 'Transfer', 'wd'] = False
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == False), 'wd'] = False
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == True ), 'wd'] = True

print('calendar')
print(calendar)


##### 3. 데이터 로드 후 type 설정
df_train = pd.read_csv(path + 'train.csv',
                       usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
                       dtype={'store_nbr': 'category', 'family': 'category', 'sales': 'float64', 'onpromotion': 'float64'},
                       parse_dates=['date'], infer_datetime_format=True)
df_test = pd.read_csv(path + 'test.csv',
                       usecols=['store_nbr', 'family', 'date', 'onpromotion'],
                       dtype={'store_nbr': 'category', 'family': 'category', 'onpromotion': 'float64'},
                       parse_dates=['date'], infer_datetime_format=True)
##### oil
data_oil = pd.read_csv(path + 'oil.csv', parse_dates=['date'], infer_datetime_format=True, index_col='date')
df_train = pd.merge(df_train, data_oil, left_on='date', right_on='date', how='left')
df_test = pd.merge(df_test, data_oil, left_on='date', right_on='date', how='left')

##### 결측값 제거
# df_train = df_train.dropna(axis=0)
df_train['dcoilwtico'] = df_train['dcoilwtico'].fillna(0)
df_train = df_train.reset_index(drop=True)

# df_test = df_test.dropna(axis=0)
df_test['dcoilwtico'] = df_test['dcoilwtico'].fillna(0)
df_test = df_test.reset_index(drop=True)

print('df_train')
print(df_train)
origin_train = df_train
origin_test = df_test
# origin_train.to_csv('./results/1_origin_train.csv', index=True)

##### 원핫인코딩
ohe = OneHotEncoder(sparse=False)
# fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
train_cat = ohe.fit_transform(origin_train[['family']])
origin_train = pd.concat([origin_train.drop(columns=['family']),
           pd.DataFrame(train_cat, columns=['family_' + col for col in ohe.categories_[0]])], axis=1)

test_cat = ohe.transform(origin_test[['family']])
origin_test = pd.concat([origin_test.drop(columns=['family']),
           pd.DataFrame(test_cat, columns=['family_' + col for col in ohe.categories_[0]])], axis=1)

print('## origin_train')
print(origin_train)
# origin_train.to_csv('./results/2_origin_train.csv', index=True)


##### 4. Timestamp를 Period(1일)로 변환 후 인덱스 설정 및 정렬
df_train.date = df_train.date.dt.to_period('D')
df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

df_test.date = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
print('df_train_2')
print(df_train)


##### 5. date fourier
y = df_train.unstack(['store_nbr', 'family'])
fourier = CalendarFourier(freq='W', order=4)
dp = DeterministicProcess(index=y.index,
                          constant=False,
                          order=1,
                          seasonal=False,
                          additional_terms=[fourier],
                          drop=True)

y_test = df_test.unstack(['store_nbr', 'family'])
fourier_test = CalendarFourier(freq='W', order=4)
dp_test = DeterministicProcess(index=y_test.index,
                          constant=False,
                          order=1,
                          seasonal=False,
                          additional_terms=[fourier_test],
                          drop=True)

X = dp.in_sample()
X = X.reset_index()
X = X.drop(columns=['trend'])
X.to_csv('./results/XX.csv', index=True)
XX = pd.read_csv('./results/XX.csv', parse_dates=['date'], infer_datetime_format=True, index_col='date')

X_test = dp_test.in_sample()
X_test = X_test.reset_index()
X_test = X_test.drop(columns=['trend'])
X_test.to_csv('./results/XX_test.csv', index=True)
XX_test = pd.read_csv('./results/XX_test.csv', parse_dates=['date'], infer_datetime_format=True, index_col='date')

print('## origin_train')
print(origin_train)
print('## XX')
print(XX)

###### 6. left joing
TRAIN = pd.merge(origin_train, XX, left_on='date', right_on='date', how='left')
TRAIN = TRAIN.drop(columns=['date', 'store_nbr'])
print('TRAIN')
print(TRAIN)

TEST = pd.merge(origin_test, XX_test, left_on='date', right_on='date', how='left')
TEST = TEST.drop(columns=['date', 'store_nbr'])
print('TEST')
print(TEST)
TEST = TEST.astype(dtype='float64')
test_data = TEST.to_numpy()
# TEST.to_csv('./results/TEST.csv', index=True)

train = TRAIN.drop(columns='sales')
# train.to_csv('./results/train.csv', index=True)
target = TRAIN['sales']

print('train')
print(train)
print('train 2')
print(train)
print('target')
print(target)
# train.to_csv('./results/_train.csv', index=True)
# target.to_csv('./results/_target.csv', index=True)

##### 타입 변경
train = train.astype(dtype='float64')
target = target.astype(dtype='float64')

##### convert to numpy
train = train.to_numpy()
target = target.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(train, target, train_size=0.8, random_state=5)
print('type(x_train)', type(x_train))
print('x_train')
print(x_train)
print('x_train.dtype')
print(x_train.dtype)

sample_submission = pd.read_csv('./input/store-sales-time-series-forecasting/sample_submission.csv')

##########################
def relu(x):
    relu = []
    for i in x:
        if i < 0:
            relu.append(0)
        else:
            relu.append(i)
    return relu

# ##### XG
# print('6. XGBoostRegressor')
#
# import xgboost as xgb
#
# XG = xgb.XGBRegressor(objective = 'reg:squarederror' , learning_rate = 0.1,
#                 max_depth = 10, n_estimators = 100).fit(x_train, y_train)
# y_pred_XG = XG.predict(x_test)
#
# # print('y_test', y_test)
# # print('y_pred_XG', y_pred_XG)
# # print('relu(y_pred_XG)', relu(y_pred_XG))
# MSLE = mean_squared_log_error(y_test, relu(y_pred_XG))
# RMSLE = np.sqrt(mean_squared_log_error(y_test, relu(y_pred_XG)))
# print('VAL MSLE : ', MSLE)
# print('VAL RMSLE : ', RMSLE)
#
# sub = XG.predict(TEST)
# # print('sub', sub)
# # print('relu(sub)', relu(sub))
# sample_submission['sales'] = relu(sub)
# sample_submission.to_csv('./results/submission.csv', index=False)



import tensorflow as tf
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

tf.keras.backend.set_floatx('float64')

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(256, input_shape=(42, ), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dense(256, input_shape=(42, ), kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# ########
# model.add(tf.keras.layers.Dense(32, input_shape=(42, )))
# model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(1))

model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=32, input_shape=(15, 1)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
])

model.compile(optimizer=Adam(lr=0.01), loss=root_mean_squared_error,
              metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))
# model.compile(optimizer='rmsprop', loss=root_mean_squared_error,
#               metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, batch_size=4096, epochs=30, validation_split=0.2)

tf.keras.utils.plot_model(model,
                          to_file='model_plot.png',
                          show_shapes=True
                          )
my_y_pred = model.predict(x_test)
# my_y_pred = model.predict(x_test.to_numpy()).reshape((1, 15, 1))
print('my_y_pred', my_y_pred)
MY_MSLE = mean_squared_log_error(y_test, relu(my_y_pred))
MY_RMSLE = np.sqrt(mean_squared_log_error(y_test, relu(my_y_pred)))
print('MY_MSLE : ', MY_MSLE)
print('MY_RMSLE : ', MY_RMSLE)

# print('######### x_test')
# print(x_test)
# print('######### test_data')
# print(test_data)

my_sub = model.predict(test_data)
print('my_sub', my_sub)


sample_submission['sales'] = relu(my_sub)
sample_submission.to_csv('./results/submission.csv', index=False)