# https://www.kaggle.com/competitions/psnr-iqa
import collections

from sklearn.impute import KNNImputer

is_train = True

w1 = 1
w2 = 1
w3 = 1
w4 = 1

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
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.ops.variable_scope import get_variable
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier


# def label_encoding(in_df, column_list):
#     out_df = in_df.copy()
#
#     for col in column_list:
#         le = LabelEncoder()
#         value_data = out_df[col].values
#         new_data = le.fit_transform(value_data)
#
#         out_df[col] = new_data
#
#     return out_df
#
#
# def onehot_encoding(in_df, column_list):
#     out_df = in_df.copy()
#
#     for col in column_list:
#         new_data = pd.get_dummies((out_df[col]))
#
#         for i, name in zip(range(len(out_df)), out_df.columns):
#             if name == col:
#                 for n in new_data.columns:
#                     out_df.insert(i, str(col)+'_'+str(n), new_data[n])
#                 break
#
#         out_df.drop(col, axis=1, inplace=True)
#
#     return out_df


def standard_scaling(in_train_df, in_val_df, in_test_df, column_list=None):
    if column_list is None:
        column_list = in_train_df.columns
    for col in column_list:
        # std = StandardScaler()
        std = MinMaxScaler()
        new_train_data = std.fit_transform(in_train_df[[col]])
        in_train_df[col] = new_train_data

        new_val_data = std.transform(in_val_df[[col]])
        in_val_df[col] = new_val_data

        new_test_data = std.transform(in_test_df[[col]])
        in_test_df[col] = new_test_data

    return in_train_df, in_val_df, in_test_df


def pre_processing(train_file_name, test_file_name, val_size=0.):
    total_df = pd.read_csv(train_file_name)
    print('total size', len(total_df))
    print('total_df["compliance_2021"].value_counts()')
    print(total_df['compliance_2021'].value_counts())

    test_df = pd.read_csv(test_file_name)
    print('test_size', len(test_df))

    # y 저장
    total_y = total_df['compliance_2021'].to_numpy()

    # id제거
    total_df.drop('compliance_2021', axis=1, inplace=True)
    total_df.drop('station_id', axis=1, inplace=True)
    test_df.drop('station_id', axis=1, inplace=True)

    train_df = total_df.copy()
    val_df = total_df.copy()
    y_train = total_y.copy()
    y_val = total_y.copy()
    if val_size != 0.:
        train_df, val_df, y_train, y_val = train_test_split(total_df, total_y, test_size=val_size,
                                                            stratify=total_y, random_state=144)
    print('train_size', len(train_df))
    print('collections.Counter(y_train)')
    print(collections.Counter(y_train))
    print('val_size', len(val_df))
    print('collections.Counter(y_val)')
    print(collections.Counter(y_val))

    print('##### 디스크립션')
    print(train_df.describe())
    print('##### 결측치 확인(개수)')
    print('========== columns ==========')
    print(train_df.isnull().sum(), type(train_df.isnull().sum()))
    print('##### 고윳값 확인')
    for col in train_df.columns:
        print('=============== ' + col + ' ===============')
        print(train_df[col].unique())
    print('##### 고윳값 개수 확인')
    print(train_df.nunique())
    print('##### 불필요한 칼럼 제거')
    train_df.drop(['compliance_2019', 'compliance_2020'], axis=1, inplace=True)
    val_df.drop(['compliance_2019', 'compliance_2020'], axis=1, inplace=True)
    test_df.drop(['compliance_2019', 'compliance_2020'], axis=1, inplace=True)
    # remove_columns = ['Aluminium_2019', 'Aluminium_2020', 'Boron_2019', 'Boron_2020', 'Chloride_2019', 'Chloride_2020',
    #                   'Coli-like-bacteria-Colilert_2019', 'Coli-like-bacteria-Colilert_2020',
    #                   'Color-Pt-Co-unit_2019', 'Color-Pt-Co-unit_2020',
    #                   'Escherichia-coli-Colilert_2019', 'Escherichia-coli-Colilert_2020',
    #                   'Fluoride_2019', 'Fluoride_2020', 'Nitrate_2019', 'Nitrate_2020', 'Nitrite_2019', 'Nitrite_2020',
    #                   'Oxidability_2019', 'Oxidability_2020', 'Smell-ball-units_2019', 'Smell-ball-units_2020',
    #                   # 'Sodium_2019', 'Sodium_2020', 'Sulphate_2019', 'Sulphate_2020',
    #                   'Taste-ball-units_2019', 'Taste-ball-units_2020',
    #                   'Escherichia-coli_2019', 'Escherichia-coli_2020']
    # train_df.drop(remove_columns, axis=1, inplace=True)
    print('##### 결측치 대체')
    imputer = KNNImputer(n_neighbors=3)
    train_np = imputer.fit_transform(train_df.to_numpy())
    train_df = pd.DataFrame(train_np, columns=train_df.columns)
    val_np = imputer.transform(val_df.to_numpy())
    val_df = pd.DataFrame(val_np, columns=val_df.columns)
    test_np = imputer.transform(test_df.to_numpy())
    test_df = pd.DataFrame(test_np, columns=test_df.columns)

    print('========== columns ==========')
    print(train_df.isnull().sum(), type(train_df.isnull().sum()))

    # train_df = train_df.fillna(train_df.interpolate())
    # train_df = train_df.fillna(train_df.mean())
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    test_df = test_df.fillna(0)
    print('##### 인코딩')
    # # train_df = standard_scaling(train_df, train_df.columns.difference(['compliance_2019', 'compliance_2020']))
    # train_df = standard_scaling(train_df,
    #                             ['Colony-count-at-22-C_2019', 'Colony-count-at-22-C_2020',
    #                              'Electrical-conductivity_2019', 'Electrical-conductivity_2020',
    #                              'Iron_2019', 'Iron_2020', 'Manganese_2019', 'Manganese_2020',
    #                              'Sodium_2019', 'Sodium_2020', 'Sulphate_2019', 'Sulphate_2020'])
    train_df, val_df, test_df = standard_scaling(train_df, val_df, test_df)

    # print('##### 데이터 증강')
    # df_columns = train_df.columns
    # for i in range(0, len(df_columns) - 2, 2):
    #     column_name = df_columns[i].split('_')[0]
    #
    #     train_df[column_name + '_sum'] = train_df.iloc[:, [i, i + 1]].sum(axis=1)
    #     train_df[column_name + '_min'] = train_df.iloc[:, [i, i + 1]].min(axis=1)
    #     train_df[column_name + '_max'] = train_df.iloc[:, [i, i + 1]].max(axis=1)
    #     train_df[column_name + '_std'] = train_df.iloc[:, [i, i + 1]].std(axis=1)
    #     train_df[column_name + '_mad'] = train_df.iloc[:, [i, i + 1]].mad(axis=1)
    #     train_df[column_name + '_mean'] = train_df.iloc[:, [i, i + 1]].mean(axis=1)
    #     # train_df[column_name + '_kurt'] = train_df.iloc[:, [i, i + 1]].kurt(axis=1)

    train_df.to_csv('preprocessed_train_data.csv', index=False)
    val_df.to_csv('preprocessed_val_data.csv', index=False)
    test_df.to_csv('preprocessed_test_data.csv', index=False)
    print('return train_df.columns')
    print(train_df.columns)

    return train_df, val_df, y_train, y_val, test_df


def cross_validation(model, X_train_fold, y_train_fold, splits=5, name='model'):
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    n_iter = 0
    for train_idx, val_idx in kfold.split(X_train_fold, y_train_fold):
        kfold_train_X, kfold_val_X = X_train_fold[train_idx], X_train_fold[val_idx]
        kfold_train_y, kfold_val_y = y_train_fold[train_idx], y_train_fold[val_idx]

        model.fit(kfold_train_X, kfold_train_y)
        n_iter += 1

        y_pred = model.predict(kfold_train_X)
        y_pred_val = model.predict(kfold_val_X)
        # y_pred_proba = model.predict_proba(kfold_train_X)
        # y_pred_val_proba = model.predict_proba(kfold_val_X)

        # print(n_iter, 'KFOLD %s Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
        #       (name, accuracy_score(kfold_train_y, y_pred), precision_score(kfold_train_y, y_pred, average='micro'),
        #        recall_score(kfold_train_y, y_pred, average='micro'),
        #        f1_score(kfold_train_y, y_pred, average='micro')))
        # print(n_iter, 'KFOLD %s Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
        #       (name, accuracy_score(kfold_val_y, y_pred_val),
        #        precision_score(kfold_val_y, y_pred_val, average='micro'),
        #        recall_score(kfold_val_y, y_pred_val, average='micro'),
        #        f1_score(kfold_val_y, y_pred_val, average='micro')))

        # print('kfold_val_y')
        # print(kfold_val_y)
        # print('y_pred_' + name)
        # print(y_pred_val)
        # print(n_iter, 'end..')

    return model

########## 데이터 전처리 및 분리
X_train, X_val, y_train, y_val, X_test = pre_processing('train.csv', 'test.csv', 0.3)

# ########## 데이터 정규성 확인
# import seaborn as sns
# for col in X_train.columns:
#     sns.distplot(X_train[col])
#     plt.title(col)
#     plt.show()

# X_train, X_val, y_train, y_val = X, X, y, y

# DL data
X_train_dl = X_train.copy()
X_val_dl = X_val.copy()
X_test_dl = X_test.copy()

########## 증식 테스트
y_train_size = len(y_train)
print('SMOTE pre len(y_train)', y_train_size)
from imblearn.over_sampling import SMOTE
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", X_train.shape, y_train.shape)
print('SMOTE 적용 전 값의 분포 :\n', pd.Series(y_train).value_counts())
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print('SMOTE post y_train', len(y_train))
X_train = X_train[:y_train_size + 30]
y_train = y_train[:y_train_size + 30]
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', X_train.shape, y_train.shape)
print('SMOTE 적용 후 값의 분포 :\n', pd.Series(y_train).value_counts())
pd.DataFrame(X_train, columns=X_train.columns).to_csv('preprocessed_smote_train_data.csv', index=False)


print('@@@@ X_train.shape, X_val.shape, X_test.shape', X_train.shape, X_val.shape, X_test.shape)
print('@@@@ X_train', X_train)
print('@@@@ X_val', X_val)
print('@@@@ x_test', X_test)


# ########## PCA 테스트
from sklearn.decomposition import PCA
pca_n = 50
pcn_columns = list(range(pca_n))
pca = PCA(n_components=pca_n) # 주성분을 몇개로 할지 결정
printcipal_components_train = pca.fit_transform(X_train)
principalDf_train = pd.DataFrame(data=printcipal_components_train, columns=pcn_columns)
principalDf_train.to_csv('preprocessed_pca_train_data.csv', index=False)
# print('principalDf.head()', principalDf.head())
print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
print('sum(pca.explained_variance_ratio_)', sum(pca.explained_variance_ratio_))

printcipal_components_val = pca.transform(X_val)
principalDf_val = pd.DataFrame(data=printcipal_components_val, columns=pcn_columns)
principalDf_val.to_csv('preprocessed_pca_val_data.csv', index=False)

printcipal_components_test = pca.transform(X_test)
principalDf_test = pd.DataFrame(data=printcipal_components_test, columns=pcn_columns)
principalDf_test.to_csv('preprocessed_pca_test_data.csv', index=False)

X_train = principalDf_train.to_numpy()
X_val = principalDf_val.to_numpy()
X_test = principalDf_test.to_numpy()
print('@@@@ PCA X_train, X_train.shape', X_train, X_train.shape)
print('@@@@ PCA X_val, X_val.shape', X_val, X_val.shape)
print('@@@@ PCA X_test, X_test.shape', X_test, X_test.shape)

### cross validation data
X_train_fold = np.concatenate((X_train, X_val), axis=0)
y_train_fold = np.concatenate((y_train, y_val), axis=0)


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
    # xg.fit(X_train, y_train)
    xg = cross_validation(xg, X_train_fold, y_train_fold, 5, 'xg')
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
    # gb.fit(X_train, y_train)
    gb = cross_validation(gb, X_train_fold, y_train_fold, 5, 'gb')
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
    # rfc.fit(X_train, y_train)
    rfc = cross_validation(rfc, X_train_fold, y_train_fold, 5, 'rfc')
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
lgb = LGBMClassifier()
# lgb.fit(X_train, y_train)
lgb = cross_validation(lgb, X_train_fold, y_train_fold, 5, 'lgb')
joblib.dump(lgb, 'lgb_cls_model.pkl')

lgb = joblib.load('lgb_cls_model.pkl')
y_pred_lgb = lgb.predict(X_train)
y_pred_lgb_val = lgb.predict(X_val)
y_pred_lgb_proba = lgb.predict_proba(X_train)
y_pred_lgb_val_proba = lgb.predict_proba(X_val)

print('LGB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, y_pred_lgb), precision_score(y_train, y_pred_lgb, average='micro'),
       recall_score(y_train, y_pred_lgb, average='micro'), f1_score(y_train, y_pred_lgb, average='micro')))
print('LGB Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_lgb_val), precision_score(y_val, y_pred_lgb_val, average='micro'),
       recall_score(y_val, y_pred_lgb_val, average='micro'), f1_score(y_val, y_pred_lgb_val, average='micro')))


###### gbr
gbr = GradientBoostingClassifier(random_state=0)
# gbr.fit(X_train, y_train)
gbr = cross_validation(gbr, X_train_fold, y_train_fold, 5, 'gbr')
y_pred_gbr = gbr.predict(X_val)

print('GBR Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_gbr), precision_score(y_val, y_pred_gbr, average='micro'),
       recall_score(y_val, y_pred_gbr, average='micro'), f1_score(y_val, y_pred_gbr, average='micro')))

###### ridge
ridge = RidgeClassifier()
# ridge.fit(X_train, y_train)
ridge = cross_validation(ridge, X_train_fold, y_train_fold, 5, 'ridge')
y_pred_ridge = ridge.predict(X_train)
y_pred_ridge_val = ridge.predict(X_val)

print('RIDGE Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, y_pred_ridge), precision_score(y_train, y_pred_ridge, average='micro'),
       recall_score(y_train, y_pred_ridge, average='micro'), f1_score(y_train, y_pred_ridge, average='micro')))
print('RIDGE Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_ridge_val), precision_score(y_val, y_pred_ridge_val, average='micro'),
       recall_score(y_val, y_pred_ridge_val, average='micro'), f1_score(y_val, y_pred_ridge_val, average='micro')))

###### dt
dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
dt = cross_validation(dt, X_train_fold, y_train_fold, 5, 'dt')
y_pred_dt = dt.predict(X_val)

print('DT Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_dt), precision_score(y_val, y_pred_dt, average='micro'),
       recall_score(y_val, y_pred_dt, average='micro'), f1_score(y_val, y_pred_dt, average='micro')))

###### gnb
gnb = GaussianNB()
# gnb.fit(X_train, y_train)
gnb = cross_validation(gnb, X_train_fold, y_train_fold, 5, 'gnb')
y_pred_gnb = gnb.predict(X_val)

print('GNB Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, y_pred_gnb), precision_score(y_val, y_pred_gnb, average='micro'),
       recall_score(y_val, y_pred_gnb, average='micro'), f1_score(y_val, y_pred_gnb, average='micro')))

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

##### final
# # final_pred = (0.4 * y_pred) + (0.4 * y_pred_gbr) + (0.1 * y_pred_ridge) + (0.1 * y_pred_lgb)
# # final_pred = (0.25 * y_pred_xgb) + (0.3 * y_pred_lgb) + (0.35 * y_pred_rfc) + (0.1 * y_pred_dt)
# final_pred = (w1 * y_pred_xgb) + (w2 * y_pred_rfc) + (w3 * y_pred_lgb) + (w4 * y_pred_ridge)
# final_pred = np.round(np.round(final_pred)).astype(np.int16)
# print('FINAL(XGB, RFC, LGB, RIDGE) Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, final_pred), precision_score(y_train, final_pred, average='micro'),
#        recall_score(y_train, final_pred, average='micro'), f1_score(y_train, final_pred, average='micro')))
#
# final_val_pred = (w1 * y_pred_xgb_val) + (w2 * y_pred_rfc_val) + (w3 * y_pred_lgb_val) + (w4 * y_pred_ridge_val)
# final_val_pred = np.round(np.round(final_val_pred)).astype(np.int16)
# print('FINAL(XGB, RFC, LGB, RIDGE) Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, final_val_pred), precision_score(y_val, final_val_pred, average='micro'),
#        recall_score(y_val, final_val_pred, average='micro'), f1_score(y_val, final_val_pred, average='micro')))

# # RECENTLY
# final_pred_proba = ((w1 * y_pred_xgb_proba) + (w2 * y_pred_gb_proba) + (w3 * y_pred_rfc_proba)) / (w1 + w2 + w3)
# final_pred_proba = np.argmax(final_pred_proba, axis=1)
# print('FINAL(XGB, GB, RFC) Prob Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_train, final_pred_proba), precision_score(y_train, final_pred_proba, average='micro'),
#        recall_score(y_train, final_pred_proba, average='micro'), f1_score(y_train, final_pred_proba, average='micro')))
#
# final_val_pred_proba = ((w1 * y_pred_xgb_val_proba) + (w2 * y_pred_gb_val_proba) + (w3 * y_pred_rfc_val_proba)) / (w1 + w2 + w3)
# final_val_pred_proba = np.argmax(final_val_pred_proba, axis=1)
# print('FINAL(XGB, GB, RFC) Prob Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
#       (accuracy_score(y_val, final_val_pred_proba), precision_score(y_val, final_val_pred_proba, average='micro'),
#        recall_score(y_val, final_val_pred_proba, average='micro'), f1_score(y_val, final_val_pred_proba, average='micro')))

# # final 지정
final_pred_proba = y_pred_xgb_proba
final_pred_proba = np.argmax(final_pred_proba, axis=1)
print('FINAL(XGB) Prob Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, final_pred_proba), precision_score(y_train, final_pred_proba, average='micro'),
       recall_score(y_train, final_pred_proba, average='micro'), f1_score(y_train, final_pred_proba, average='micro')))

final_val_pred_proba = y_pred_xgb_val_proba
final_val_pred_proba = np.argmax(final_val_pred_proba, axis=1)
print('FINAL(XGB) Prob Val Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, final_val_pred_proba), precision_score(y_val, final_val_pred_proba, average='micro'),
       recall_score(y_val, final_val_pred_proba, average='micro'), f1_score(y_val, final_val_pred_proba, average='micro')))


# # ####
vc = VotingClassifier(estimators=[('xg', xg), ('rfc', rfc), ('lgb', lgb)], voting='hard')
vc = vc.fit(X_train, y_train)
final_pred_vc = vc.predict(X_train)
final_pred_vc_val = vc.predict(X_val)
print('FINAL VOTING(XGB, RFC, LGB) VC Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_train, final_pred_vc), precision_score(y_train, final_pred_vc, average='micro'),
       recall_score(y_train, final_pred_vc, average='micro'), f1_score(y_train, final_pred_vc, average='micro')))
print('FINAL VOTING VAL (XGB, RFC, LGB) VC Accuracy : %.4f, Precision : %.4f, Recall : %.4f, F1 : %.4f' %
      (accuracy_score(y_val, final_pred_vc_val), precision_score(y_val, final_pred_vc_val, average='micro'),
       recall_score(y_val, final_pred_vc_val, average='micro'), f1_score(y_val, final_pred_vc_val, average='micro')))

#### sen spe fpr acc
# tn, fp, fn, tp = confusion_matrix(y_train, final_pred_vc).ravel()
# tn, fp, fn, tp = confusion_matrix(y_val, final_pred_vc_val).ravel()
tn, fp, fn, tp = confusion_matrix(y_val, final_val_pred_proba).ravel()
print('Confusion Matrix')
print('TP', tp)
print('TN', tn)
print('FP', fp)
print('FN', fn)

sen = None
spe = None
ppv = None
fpr = None
if tp == 0 or (tp + fn) == 0:
    sen = 0
else:
    sen = tp / (tp + fn)

if tn == 0 or (tn + fp) == 0:
    spe = 0
else:
    spe = tn / (tn + fp)

if tp == 0 or (tp + fp) == 0:
    ppv = 0
else:
    ppv = tp / (tp + fp)

if fp == 0 or (fp + tn) == 0:
    fpr = 0
else:
    fpr = fp / (fp + tn)
acc = (tp + tn) / (tp + tn + fp + fn)
print('Sensitivity', sen)
print('Specificity', spe)
print('FPR', fpr)
print('Accuracy', acc)


# print('weights w1, w2, w3', w1, w2, w3)
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
# results_rfc = rfc.predict(X_test)
# # results_gb = gb.predict(X_test)
# # results_final = (0.25 * results_xgb) + (0.3 * results_lgb) + (0.35 * results_rfc) + (0.1 * results_dt)
# # results_final = ((w1 * results_xgb) + (w2 * results_gb) + (w3 * results_rfc)) / (w1 + w2 + w3)
# results_final = ((w1 * results_xgb) + (w2 * results_lgb) + (w3 * results_rfc) + (w4 * results_ridge)) / (w1 + w2 + w3 + w4)
#### VOTING
# results_final = vc.predict(X_test)
# # 특정 모델 출력 설정
results_final = results_xgb
# print('results of xgb', y_pred_xgb_val_proba)
# print('results of rfc', y_pred_rfc_val_proba)
# print('results of lgb', y_pred_lgb_val_proba)
print('xgb results_final', results_final)

my_submission = pd.read_csv('sample_submission.csv')
my_submission['compliance_2021'] = results_final
my_submission.to_csv('my_submission.csv', index=False)


y_0 = 0
y_1 = 0
for i in results_final:
    if i == 0:
        y_0 += 1
    elif i == 1:
        y_1 += 1
print('y_00, y_11', y_0, y_1)




################### DL
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam

model = Sequential([
    # # Input(shape=(218,)),
    # Input(shape=(20,)),
    # # Input(shape=(114,)),
    # Dense(64, activation='relu'),
    # Dropout(0.3),
    # Dense(128, activation='relu'),
    # Dropout(0.3),
    # Dense(256, activation='relu'),
    # Dropout(0.3),
    # Dense(1024, activation='relu'),
    # Dropout(0.3),
    # Dense(1, activation='sigmoid')

    Input(shape=(2, 20)),
    LSTM(10),
    Flatten(),
    Dense(8, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

epochs = 3000

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.20,
#                               patience=3, min_lr=0.0009)

# model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

tf.keras.utils.plot_model(model,
                          to_file='model_plot.png',
                          show_shapes=True
                          )

# history = model.fit(X_train, y_train, epochs=epochs, callbacks=[es, mc], batch_size=64, validation_split=0.2)


X_train_dl = X_train_dl.to_numpy()
# print('@@@@@ DL  X_train_dl', X_train_dl, X_train_dl.shape)
X_val_dl = X_val_dl.to_numpy()
X_test_dl = X_test_dl.to_numpy()

total_data_set = []
data_set = []
for i in X_train_dl:
    data_set.append(i[0::2])
    data_set.append(i[1::2])
    total_data_set.append(data_set)
    data_set = []
X_train_dl = np.array(total_data_set)

total_data_set = []
data_set = []
for i in X_val_dl:
    data_set.append(i[0::2])
    data_set.append(i[1::2])
    total_data_set.append(data_set)
    data_set = []
X_val_dl = np.array(total_data_set)

total_data_set = []
data_set = []
for i in X_test_dl:
    data_set.append(i[0::2])
    data_set.append(i[1::2])
    total_data_set.append(data_set)
    data_set = []
X_test_dl = np.array(total_data_set)

# print('@@@@@ DL X_train_dl', X_train_dl, X_train_dl.shape)
# print('@@@@@ DL X_val_dl', X_val_dl, X_val_dl.shape)
# print('@@@@@ DL X_test_dl', X_test_dl, X_test_dl.shape)

# pca_n = 10
pcn_columns = list(range(pca_n))
pca = PCA(n_components=pca_n) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(X_train_dl[:, 0])
principalDf_train_1 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
printcipalComponents = pca.transform(X_val_dl[:, 0])
principalDf_val_1 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
printcipalComponents = pca.transform(X_test_dl[:, 0])
principalDf_test_1 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
# print('@@ principalDf.head()', principalDf_train_1.head())
print('@@ pca.explained_variance_ratio_', pca.explained_variance_ratio_)
print('@@ sum(pca.explained_variance_ratio_)', sum(pca.explained_variance_ratio_))
X_train_dl_1 = principalDf_train_1.to_numpy()
X_val_dl_1 = principalDf_val_1.to_numpy()
X_test_dl_1 = principalDf_test_1.to_numpy()

pcn_columns = list(range(pca_n))
pca = PCA(n_components=pca_n) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(X_train_dl[:, 1])
principalDf_train_2 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
printcipalComponents = pca.transform(X_val_dl[:, 1])
principalDf_val_2 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
printcipalComponents = pca.transform(X_test_dl[:, 1])
principalDf_test_2 = pd.DataFrame(data=printcipalComponents, columns=pcn_columns)
# print('@@@ principalDf.head()', principalDf_train_2.head())
print('@@@ pca.explained_variance_ratio_', pca.explained_variance_ratio_)
print('@@@ sum(pca.explained_variance_ratio_)', sum(pca.explained_variance_ratio_))
X_train_dl_2 = principalDf_train_2.to_numpy()
X_val_dl_2 = principalDf_val_2.to_numpy()
X_test_dl_2 = principalDf_test_2.to_numpy()

X_train_dl = np.stack([X_train_dl_1, X_train_dl_2], 1)
X_val_dl = np.stack([X_val_dl_1, X_val_dl_2], 1)
X_test_dl = np.stack([X_test_dl_1, X_test_dl_2], 1)


history = model.fit(X_train_dl, y_train, epochs=epochs, callbacks=[es, mc], batch_size=64,
                    validation_data=(X_val_dl, y_val))

model = load_model('best_model.h5')

results_dl = model.predict(X_test_dl)

results_dl_converted = []
for dl in results_dl:
    if dl > 0.5:
      results_dl_converted.append(1)
    else:
      results_dl_converted.append(0)
results_dl_converted = np.array(results_dl_converted)
# print('results_dl_converted', results_dl_converted)
# print('X_train', X_train, X_train.shape)
# print('X_val', X_val, X_val.shape)
# print('X_test', X_test, X_test.shape)
y_0 = 0
y_1 = 0
for i in results_dl_converted:
    if i == 0:
        y_0 += 1
    elif i == 1:
        y_1 += 1
print('y_00, y_11', y_0, y_1)

my_submission = pd.read_csv('sample_submission.csv')
my_submission['compliance_2021'] = results_dl_converted
my_submission.to_csv('my_submission_.csv', index=False)

