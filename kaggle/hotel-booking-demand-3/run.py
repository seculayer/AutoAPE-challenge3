# https://www.kaggle.com/competitions/psnr-iqa

is_train = True

w1 = 1
w2 = 1
w3 = 1

########## 하이퍼 파라미터 + GBR, XBGR, LGBMR, RFR
########## https://dining-developer.tistory.com/23
import joblib
import numpy as np
import os
import cv2
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.ops.variable_scope import get_variable
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split, KFold
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier


def label_encoding(in_df, column_list):
    out_df = in_df.copy()

    for col in column_list:
        le = LabelEncoder()
        value_data = out_df[col].values
        new_data = le.fit_transform(value_data)

        out_df[col] = new_data

    return out_df


def onehot_encoding(in_df, column_list):
    out_df = in_df.copy()

    for col in column_list:
        new_data = pd.get_dummies((out_df[col]))

        for i, name in zip(range(len(out_df)), out_df.columns):
            if name == col:
                for n in new_data.columns:
                    out_df.insert(i, str(col)+'_'+str(n), new_data[n])
                break

        out_df.drop(col, axis=1, inplace=True)

    return out_df


def standard_scaling(in_df, column_list):
    out_df = in_df.copy()

    for col in column_list:
        std = StandardScaler()
        std.fit(out_df[[col]])
        new_data = std.transform(out_df[[col]])

        out_df[col] = new_data

    return out_df


def pre_processing(train_file_name, test_file_name):
    train_df = pd.read_csv(train_file_name)
    train_size = len(train_df)
    print('train_size', train_size)

    # y 저장
    y = train_df['is_canceled'].to_numpy()
    train_df.drop('is_canceled', axis=1, inplace=True)

    # test 로드
    test_df = pd.read_csv(test_file_name)

    # 합치기
    train_df = pd.concat([train_df, test_df], ignore_index=True)
    print('len(train_df)', len(train_df))
    # print('check train_df.loc[train_size]')
    # print(train_df.loc[train_size])
    # print('check test_df.loc[0]')
    # print(test_df.loc[0])

    print('##### 결측치 확인')
    print('========== columns ==========')
    print(train_df.isnull().sum())
    print('##### 고윳값 확인')
    for col in train_df.columns:
        print('=============== ' + col + ' ===============')
        print(train_df[col].unique())
    print('##### 고윳값 개수 확인')
    print(train_df.nunique())
    print('##### 불필요한 칼럼 제거')

    # train_df.drop('reservation_status_date', axis=1, inplace=True)
    ### 'arrival_date_week_number', 'arrival_date_day_of_month' 드롭 추가
    train_df.drop(['reservation_status_date', 'arrival_date_week_number', 'arrival_date_day_of_month'], axis=1, inplace=True)

    ######### 전처리
    new_train_df = label_encoding(train_df, ['hotel'])
    new_train_df = onehot_encoding(new_train_df,
                                   ['arrival_date_year', 'arrival_date_month',
                                    'meal', 'country', 'market_segment',
                                    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                                    'deposit_type', 'customer_type'])
    # new_train_df = standard_scaling(new_train_df,
    #                                 ['lead_time', 'arrival_date_week_number', 'stays_in_weekend_nights',
    #                                  'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations',
    #                                  'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
    #                                  'adr', 'required_car_parking_spaces', 'total_of_special_requests'])


    new_train_df.to_csv('preprocessed_data.csv', index=False)
    print('new_train_df.columns')
    print(new_train_df.columns)

    X = new_train_df


    print('train_df')
    print(train_df)
    print()
    print('new_train_df')
    print(new_train_df)

    return X[:train_size], y, X[train_size:]

########## 데이터 확인
X, y, X_test = pre_processing('train_final.csv', 'test_final.csv')

# for i in range(9):
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=144)
X_train, X_val, y_train, y_val = X, X, y, y
print('@@@@ X_train.shape, X_val.shape, X_test.shape', X_train.shape, X_val.shape, X_test.shape)
print('@@@@ X_train', X_train)

###### xgb
colsample_bytree = 1
gamma = 1.5
learning_rate = 0.0001
max_depth = 32
n_estimators = 100000
reg_alpha = 2.1
reg_lambda = 2.3
subsample = 0.5
min_child_weight = 1

if is_train:
    xg = XGBClassifier(gamma=0.5, learning_rate=0.03, max_depth=32, n_estimators=1000)
    # xg.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)])
    xg.fit(X_train, y_train)
    # xg = xgb.XGBRegressor(colsample_bytree=colsample_bytree, gamma=gamma,
    #                       learning_rate=learning_rate, max_depth=max_depth,
    #                       n_estimators=n_estimators,
    #                       reg_alpha=reg_alpha, reg_lambda=reg_lambda,
    #                       subsample=subsample, silent=1, min_child_weight=min_child_weight,
    #                       random_state=7, nthread=-1)
    # xg.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', early_stopping_rounds=100)
    joblib.dump(xg, 'xg_cls_model.pkl')

xg = joblib.load('xg_cls_model.pkl')
y_pred_xgb = xg.predict(X_train)
y_pred_xgb_val = xg.predict(X_val)
y_pred_xgb_proba = xg.predict_proba(X_train)
y_pred_xgb_val_proba = xg.predict_proba(X_val)

print('#### y_pred_xgb', y_pred_xgb)
print('#### y_train', y_train)

# XGB Accuracy : 0.8788
print('XGB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, y_pred_xgb), precision_score(y_train, y_pred_xgb, average='micro'),
       recall_score(y_train, y_pred_xgb, average='micro'), f1_score(y_train, y_pred_xgb, average='micro')))
print('XGB Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_xgb_val), precision_score(y_val, y_pred_xgb_val, average='micro'),
       recall_score(y_val, y_pred_xgb_val, average='micro'), f1_score(y_val, y_pred_xgb_val, average='micro')))


###### gb
if is_train:
    gb = GradientBoostingClassifier(n_estimators=1800,
                                    learning_rate=0.01636280613755809,
                                    max_depth=32,
                                    max_features='sqrt',
                                    min_samples_leaf=5,
                                    min_samples_split=9,
                                    random_state=1,
                                    validation_fraction=0.3,
                                    n_iter_no_change=100)
    gb.fit(X_train, y_train)
    joblib.dump(gb, 'gb_cls_model.pkl')

gb = joblib.load('gb_cls_model.pkl')
y_pred_gb = gb.predict(X_train)
y_pred_gb_val = gb.predict(X_val)
y_pred_gb_proba = gb.predict_proba(X_train)
y_pred_gb_val_proba = gb.predict_proba(X_val)

print('GB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, y_pred_gb), precision_score(y_train, y_pred_gb, average='micro'),
       recall_score(y_train, y_pred_gb, average='micro'), f1_score(y_train, y_pred_gb, average='micro')))
print('GB Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_gb_val), precision_score(y_val, y_pred_gb_val, average='micro'),
       recall_score(y_val, y_pred_gb_val, average='micro'), f1_score(y_val, y_pred_gb_val, average='micro')))


###### rfc
if is_train:
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    joblib.dump(rfc, 'rfc_cls_model.pkl')

rfc = joblib.load('rfc_cls_model.pkl')
y_pred_rfc = rfc.predict(X_train)
y_pred_rfc_val = rfc.predict(X_val)
y_pred_rfc_proba = rfc.predict_proba(X_train)
y_pred_rfc_val_proba = rfc.predict_proba(X_val)

print('RFC Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, y_pred_rfc), precision_score(y_train, y_pred_rfc, average='micro'),
       recall_score(y_train, y_pred_rfc, average='micro'), f1_score(y_train, y_pred_rfc, average='micro')))
print('RFC Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_rfc_val), precision_score(y_val, y_pred_rfc_val, average='micro'),
       recall_score(y_val, y_pred_rfc_val, average='micro'), f1_score(y_val, y_pred_rfc_val, average='micro')))




# ###### lgb
# lgb = LGBMClassifier()
# lgb.fit(X_train, y_train)
# joblib.dump(lgb, 'lgb_cls_model.pkl')
#
# lgb = joblib.load('lgb_cls_model.pkl')
# y_pred_lgb = lgb.predict(X_train)
# y_pred_lgb_val = lgb.predict(X_val)
# y_pred_lgb_proba = lgb.predict_proba(X_train)
# y_pred_lgb_val_proba = lgb.predict_proba(X_val)
#
# print('LGB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, y_pred_lgb), precision_score(y_train, y_pred_lgb, average='micro'),
#        recall_score(y_train, y_pred_lgb, average='micro'), f1_score(y_train, y_pred_lgb, average='micro')))
# print('LGB Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_lgb_val), precision_score(y_val, y_pred_lgb_val, average='micro'),
#        recall_score(y_val, y_pred_lgb_val, average='micro'), f1_score(y_val, y_pred_lgb_val, average='micro')))


# ###### gbr
# gbr = GradientBoostingClassifier(random_state=0)
# gbr.fit(X_train, y_train)
# y_pred_gbr = gbr.predict(X_val)
#
# print('GBR Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_gbr), precision_score(y_val, y_pred_gbr, average='micro'),
#        recall_score(y_val, y_pred_gbr, average='micro'), f1_score(y_val, y_pred_gbr, average='micro')))
#
# ###### ridge
# ridge = RidgeClassifier()
# ridge.fit(X_train, y_train)
# y_pred_ridge = ridge.predict(X_val)
#
# print('RIDGE Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_ridge), precision_score(y_val, y_pred_ridge, average='micro'),
#        recall_score(y_val, y_pred_ridge, average='micro'), f1_score(y_val, y_pred_ridge, average='micro')))
#
# ###### dt
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# y_pred_dt = dt.predict(X_val)
#
# print('DT Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_dt), precision_score(y_val, y_pred_dt, average='micro'),
#        recall_score(y_val, y_pred_dt, average='micro'), f1_score(y_val, y_pred_dt, average='micro')))
#
# ###### gnb
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_gnb = gnb.predict(X_val)
#
# print('GNB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_gnb), precision_score(y_val, y_pred_gnb, average='micro'),
#        recall_score(y_val, y_pred_gnb, average='micro'), f1_score(y_val, y_pred_gnb, average='micro')))

# ###### test model
# test_model = ExtraTreesClassifier()
# test_model.fit(X_train, y_train)
# y_pred_test_model = test_model.predict(X_train)
# y_pred_val_test_model = test_model.predict(X_val)
#
# print('test_model Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, y_pred_test_model), precision_score(y_train, y_pred_test_model, average='micro'),
#        recall_score(y_train, y_pred_test_model, average='micro'), f1_score(y_train, y_pred_test_model, average='micro')))
# print('test_model Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, y_pred_val_test_model), precision_score(y_val, y_pred_val_test_model, average='micro'),
#        recall_score(y_val, y_pred_val_test_model, average='micro'), f1_score(y_val, y_pred_val_test_model, average='micro')))

# #### AutoML
# from flaml import AutoML
# automl = AutoML()
#
# automl.fit(X_train, y_train, task="cls",metric='r2', time_budget=1800)

###### final
# # final_pred = (0.4 * y_pred) + (0.4 * y_pred_gbr) + (0.1 * y_pred_ridge) + (0.1 * y_pred_lgb)
# # final_pred = (0.25 * y_pred_xgb) + (0.3 * y_pred_lgb) + (0.35 * y_pred_rfc) + (0.1 * y_pred_dt)
# final_pred = (w1 * y_pred_xgb) + (w2 * y_pred_gb) + (w3 * y_pred_rfc)
# final_pred = np.round(np.round(final_pred)).astype(np.int16)
# print('FINAL(XGB, GB, RFC) Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, final_pred), precision_score(y_train, final_pred, average='micro'),
#        recall_score(y_train, final_pred, average='micro'), f1_score(y_train, final_pred, average='micro')))
#
# final_val_pred = (w1 * y_pred_xgb_val) + (w2 * y_pred_gb_val) + (w3 * y_pred_rfc_val)
# final_val_pred = np.round(np.round(final_val_pred)).astype(np.int16)
# print('FINAL(XGB, GB, RFC) Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, final_val_pred), precision_score(y_val, final_val_pred, average='micro'),
#        recall_score(y_val, final_val_pred, average='micro'), f1_score(y_val, final_val_pred, average='micro')))

final_pred_proba = ((w1 * y_pred_xgb_proba) + (w2 * y_pred_gb_proba) + (w3 * y_pred_rfc_proba)) / (w1 + w2 + w3)
final_pred_proba = np.argmax(final_pred_proba, axis=1)
print('FINAL(XGB, GB, RFC) Prob Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, final_pred_proba), precision_score(y_train, final_pred_proba, average='micro'),
       recall_score(y_train, final_pred_proba, average='micro'), f1_score(y_train, final_pred_proba, average='micro')))

final_val_pred_proba = ((w1 * y_pred_xgb_val_proba) + (w2 * y_pred_gb_val_proba) + (w3 * y_pred_rfc_val_proba)) / (w1 + w2 + w3)
final_val_pred_proba = np.argmax(final_val_pred_proba, axis=1)
print('FINAL(XGB, GB, RFC) Prob Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, final_val_pred_proba), precision_score(y_val, final_val_pred_proba, average='micro'),
       recall_score(y_val, final_val_pred_proba, average='micro'), f1_score(y_val, final_val_pred_proba, average='micro')))

# ####
# vc = VotingClassifier(estimators=[xg, gb, rfc], voting='soft', weights=[w1, w2, w3])
# vc.fit(X_train, y_train)
# final_pred_vc = vc.predict(X_train)
# print('FINAL(XGB, GB, RFC) VC Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, final_pred_vc), precision_score(y_train, final_pred_vc, average='micro'),
#        recall_score(y_train, final_pred_vc, average='micro'), f1_score(y_train, final_pred_vc, average='micro')))

print('weights w1, w2, w3', w1, w2, w3)
# print('n_iter:', i)

# model_train_results = pd.DataFrame({'is_canceled': final_pred})
# model_train_results.to_csv('model_train_results.csv', index=False)
#
# model_val_results = pd.DataFrame({'is_canceled': final_pred})
# model_val_results.to_csv('model_val_results.csv', index=False)


################# TEST
results_xgb = xg.predict(X_test)
# results_ridge = ridge.predict(X_test)
# results_lgb = lgb.predict(X_test)
results_rfc = rfc.predict(X_test)
results_gb = gb.predict(X_test)
# results_final = (0.25 * results_xgb) + (0.3 * results_lgb) + (0.35 * results_rfc) + (0.1 * results_dt)
results_final = ((w1 * results_xgb) + (w2 * results_gb) + (w3 * results_rfc)) / (w1 + w2 + w3)
results_final = np.round(results_final).astype(np.int16)

# 특정 모델 출력 설정
results_final = np.round(results_rfc).astype(np.int16)

# my_submission = pd.DataFrame({'index':[i for i in range(23525)], 'is_canceled':results_xgb})
my_submission = pd.DataFrame({'index':[i for i in range(23525)], 'is_canceled':results_final})
my_submission.to_csv('my_submission.csv', index=False)
# my_submission.to_csv('my_submission_' + str(i) + '.csv', index=False)
