# https://www.kaggle.com/competitions/tabular-playground-series-may-2022/leaderboard

import gc
import math
import scipy
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, GRU
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Flatten, Conv1D, Add
from tensorflow.keras.layers import Reshape, Dense, Dropout
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.utils.vis_utils import plot_model
import xgboost as xgb
from xgboost import XGBClassifier

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

np.random.seed(42)
tf.random.set_seed(42)


##### Load source datasets #####
train = pd.read_csv("input/tabular-playground-series-may-2022/train.csv")
train.set_index('id', inplace=True)
print(f"train: {train.shape}")
train.head()

test = pd.read_csv("input/tabular-playground-series-may-2022/test.csv")
test.set_index('id', inplace=True)
print(f"test: {test.shape}")
test.head()


##### Feature Engineering
##### https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense
for df in [train, test]:
    for i in tqdm(range(10)):
        df[f'f_27_{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')

    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

#
continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05',
                   'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23',
                   'f_24', 'f_25', 'f_26', 'f_28']

train['f_sum']  = train[continuous_feat].sum(axis=1)
train['f_min']  = train[continuous_feat].min(axis=1)
train['f_max']  = train[continuous_feat].max(axis=1)
train['f_std']  = train[continuous_feat].std(axis=1)
train['f_mad']  = train[continuous_feat].mad(axis=1)
train['f_mean'] = train[continuous_feat].mean(axis=1)
train['f_kurt'] = train[continuous_feat].kurt(axis=1)
train.head()

#
test['f_sum']  = test[continuous_feat].sum(axis=1)
test['f_min']  = test[continuous_feat].min(axis=1)
test['f_max']  = test[continuous_feat].max(axis=1)
test['f_std']  = test[continuous_feat].std(axis=1)
test['f_mad']  = test[continuous_feat].mad(axis=1)
test['f_mean'] = test[continuous_feat].mean(axis=1)
test['f_kurt'] = test[continuous_feat].kurt(axis=1)
test.head()

print('train.columns', train.columns)
print('train.head()', train.head())
# train.to_csv("train_1.csv", index=False)

#
tfidf = TfidfVectorizer(analyzer='char').fit(train['f_27'].append(test['f_27']))   # 0~1

features = tfidf.transform(train['f_27']).toarray()
features_df = pd.DataFrame(features,
                           columns=tfidf.get_feature_names(),
                           index=train.index)

train = pd.merge(train, features_df,
                 left_index=True,
                 right_index=True)

train.drop('f_27', axis=1, inplace=True)
train.head()
print('train.columns', train.columns)
print('train.head() 2', train.head())
# train.to_csv("train_2.csv", index=False)

#
features = tfidf.transform(test['f_27']).toarray()
features_df = pd.DataFrame(features,
                           columns=tfidf.get_feature_names(),
                           index=test.index)

test = pd.merge(test, features_df,
                 left_index=True,
                 right_index=True)

test.drop('f_27', axis=1, inplace=True)
test.head()

#
features = test.columns.to_list()

qt = QuantileTransformer(n_quantiles=1500,
                         output_distribution='normal',
                         random_state=42).fit(train[features])

print('features', features)
train[features] = qt.transform(train[features])
test[features] = qt.transform(test[features])
print('train.columns', train.columns)
print('train.head() 3', train.head())
# train.to_csv("train_3.csv", index=False)

##### Helper Function
def plot_confusion_matrix(cm, classes):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()


##### Keras Model
def cosine_decay(epoch):
    if epochs > 1:
        w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
    else:
        w = 1
    return w * lr_start + (1 - w) * lr_end


#
def dnn_model():
    x_input = Input(shape=(len(features),))

    xi = Dense(units=384, activation='swish')(x_input)
    xi = BatchNormalization()(xi)
    xi = Dropout(rate=0.25)(xi)

    x = Reshape((16, 24))(xi)
    x = BatchNormalization()(x)

    # xi = Dense(units=1536, activation='swish')(x_input)
    # xi = BatchNormalization()(xi)
    # xi = Dropout(rate=0.25)(xi)
    #
    # x = Reshape((32, 48))(xi)
    # x = BatchNormalization()(x)

    x = Conv1D(filters=48, activation='swish',
               kernel_size=3, strides=2,
               padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=96, activation='swish',
               kernel_size=3, strides=2,
               padding='same')(x)
    x = BatchNormalization()(x)

    # x = Conv1D(filters=96, activation='swish',
    #            kernel_size=3, strides=2,
    #            padding='same')(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv1D(filters=192, activation='swish',
    #            kernel_size=3, strides=2,
    #            padding='same')(x)
    # x = BatchNormalization()(x)

    # LSTM 추가
    print('BatchNormalization()(x).shape', x.shape)
    x = LSTM(384, activation='relu')(x)

    x = Flatten()(x)
    x = Add()([x, xi])

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x_output = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=x_input,
                  outputs=x_output,
                  name='TPS_May22_TF_Model')
    return model


def lstm_model():
    x_input = Input(shape=(len(features),))

    xi = Dense(units=384, activation='swish')(x_input)
    xi = BatchNormalization()(xi)
    xi = Dropout(rate=0.25)(xi)

    x = Reshape((16, 24))(xi)
    x = BatchNormalization()(x)

    # LSTM 추가
    print('BatchNormalization()(x).shape', x.shape)
    x = LSTM(384, activation='relu')(x)

    x = Flatten()(x)
    x = Add()([x, xi])

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x_output = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=x_input,
                  outputs=x_output,
                  name='TPS_May22_TF_Model')
    return model


def gru_model():
    x_input = Input(shape=(len(features),))

    xi = Dense(units=384, activation='swish')(x_input)
    xi = BatchNormalization()(xi)
    xi = Dropout(rate=0.25)(xi)

    x = Reshape((16, 24))(xi)
    x = BatchNormalization()(x)

    # LSTM 추가
    print('BatchNormalization()(x).shape', x.shape)
    x = GRU(384, activation='relu')(x)

    x = Flatten()(x)
    x = Add()([x, xi])

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(units=128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)

    x_output = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=x_input,
                  outputs=x_output,
                  name='TPS_May22_TF_Model')
    return model

FOLD = 10
SEEDS = [42]

lr_start = 1e-2
lr_end = 1e-4
batch_size = 2048
verbose = 1
epochs = 50

lr = LearningRateScheduler(cosine_decay, verbose=verbose)

chk_point = ModelCheckpoint(f'./TPS_May22_TF_Model_MINE.h5',
                            monitor='val_auc', verbose=verbose,
                            save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=5)

train_x, train_y = train[features], train['target']

print('train_x.shape', train_x.shape)
test_x = test[features]


### CNN
model = dnn_model()

model.compile(optimizer=Adam(learning_rate=lr_start),
              loss="binary_crossentropy", metrics=['AUC'])

plot_model(model, to_file='model.png')
model.summary()

model.fit(
    train_x, train_y,
    # validation_data=(val_x, val_y),
    validation_split=0.2,
    epochs=epochs,
    verbose=verbose,
    batch_size=batch_size,
    callbacks=[lr, es, chk_point, TerminateOnNaN()]
    # callbacks=[lr, chk_point]
)
cnn_result = model.predict(test_x)


### LSTM
model = lstm_model()

model.compile(optimizer=Adam(learning_rate=lr_start),
              loss="binary_crossentropy", metrics=['AUC'])

plot_model(model, to_file='model.png')

model.fit(
    train_x, train_y,
    validation_split=0.2,
    epochs=epochs,
    verbose=verbose,
    batch_size=batch_size,
    callbacks=[lr, es, chk_point, TerminateOnNaN()]
)
lstm_result = model.predict(test_x)


### LightGBM
def loglikelihood(preds, dtrain):

    labels = dtrain.get_label()

    preds = 1.0/(1.0+np.exp(-preds))
    grad = (preds - labels)
    hess = (preds * (1.0-preds))

    return grad, hess


import lightgbm as lgb

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
train_ds = lgb.Dataset(train_x, label=train_y)
test_ds = lgb.Dataset(val_x, label=val_y)

params = {'learning_rate': 0.001,
          'max_depth': 16,
          'boosting': 'gbdt',
          # 'objective': 'regression',
          'objective': 'binary',
          # 'metric': 'mse',
          'metric': 'auc',
          'is_training_metric': True,
          'num_leaves': 144,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.7,
          'bagging_freq': 5,
          'seed':2018}
model = lgb.train(params, train_ds, 10000, test_ds, verbose_eval=100, early_stopping_rounds=200, fobj=loglikelihood)
gbm_result = model.predict(test_x)


# ensemble
result_list = []
for i in range(len(gbm_result)):
    p0 = 0
    p1 = 0

    if lstm_result[i] < 0.5:
        p0 += 1
    else:
        p1 += 1

    if cnn_result[i] < 0.5:
        p0 += 1
    else:
        p1 += 1

    if gbm_result[i] < 0.:
        p0 += 1
    else:
        p1 += 1

    if p0 > p1:
        result_list.append(0)
    else:
        result_list.append(1)

    print(i, ':', lstm_result[i], cnn_result[i], gbm_result[i], '/', p0, p1)

ensemble_result = np.array(result_list)


##### Create submission file
print('Create submission file')
y_p = ensemble_result
print('finish predict')

sub = pd.read_csv("input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = y_p.ravel()
sub.to_csv("results/my_submission.csv", index=False)
sub.head()



