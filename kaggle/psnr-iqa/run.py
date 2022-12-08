# https://www.kaggle.com/competitions/psnr-iqa

is_train = True

w1 = 0.7
w2 = 0.1
w3 = 0.1
w4 = 0.05
w5 = 0.05
w6 = 0.1
w7 = 0

########## 하이퍼 파라미터 + GBR, XBGR, LGBMR, RFR
########## https://dining-developer.tistory.com/23
import joblib
import numpy as np
import os
import cv2
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, \
    AdaBoostRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

import xgboost as xgb

# option
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.python.keras.saving.save import load_model

epochs = 500
batch_size = 16
lr = 0.001
end = 13210
rows = end
# rows = 1000
# np.set_printoptions(threshold=20000, linewidth=np.inf)

def MAE(p1, p2):
    return abs(p1['PSNR'].to_numpy() - p2['PSNR'].to_numpy()).mean()


def load_data(images_dir, start=0, end=rows):
    name_list = []
    image_list = []

    files = os.listdir(images_dir)

    if end is None:
        end = len(files)

    # line_number = 0
    cnt = 0
    for file in files[start:end]:
        try:
            path = os.path.join(images_dir, file)

            # # #
            # # with open(path) as myfile:
            # #     total_lines = sum(1 for line in myfile)
            # #     line_number += total_lines
            # #     print(path, total_lines, line_number)
            # # #
            #
            # image = cv2.imread(path, cv2.IMREAD_COLOR)
            #
            # # test
            # image = cv2.resize(image, (320, 240))
            # # image = image.astype(np.float32) / 64
            #
            # # if image.shape != (480, 640, 3):
            # if image.shape != (240, 320, 3):
            #     print('continue!!')
            #     continue
            #
            # image_list.append(image)
            #
            # # print(cnt, 'filename:', file, images_dir)
            cnt += 1
            # del image, file

            name_list.append(file)

        except FileNotFoundError as e:
            print('ERROR : ', e)

    print(cnt, 'images_dir:', images_dir)

    print('converting..')
    names = np.array(name_list)
    # images = np.stack(image_list)
    print('end..')

    # return names, images
    return names, ''

# train_dir = 'SBC22_IQA_dataset/SBC22_IQA_dataset/tttt'
# loading images

train_dir = 'SBC22_IQA_dataset/train'
# train_dir = '../calibration/research/data/train_input/'
test_dir = 'SBC22_IQA_dataset/test'

# name_train, X_train = load_data(train_dir, end=rows)
name_train, _ = load_data(train_dir)


# train_df = pd.read_csv('train.csv', nrows=rows)
train_df = pd.read_csv('train_full.csv', nrows=rows)
X_train_2 = train_df[train_df.columns.difference(['img_name', 'PSNR'])].to_numpy()

print('0 min, max, mean, median', np.min(X_train_2[:, 0]), np.max(X_train_2[:, 0]),
      np.mean(X_train_2[:, 0]), np.median(X_train_2[:, 0]))
print('1 min, max, mean, median', np.min(X_train_2[:, 1]), np.max(X_train_2[:, 1]),
      np.mean(X_train_2[:, 1]), np.median(X_train_2[:, 1]))

y_train = train_df['PSNR'].to_numpy()
y_train = y_train.reshape(len(y_train), 1)
y_train_ori = y_train.copy()

# Scaling
scaler = MinMaxScaler() ##########
# scaler = StandardScaler() ##########
scaler = scaler.fit(y_train) ##########
print('123 y_train', y_train.shape)
y_train = scaler.transform(y_train) ##########

y_train = y_train * 1000.0 ##########
print('^^^^^^^^^^ y_train')
print(y_train)


# ### train_feature_scaling
# test_scaler_list = []
# for i in range(X_train_2.shape[1]):
#     test_scaler = StandardScaler()
#     test_scaler = test_scaler.fit(X_train_2[:, i].reshape(len(X_train_2[:, i]), 1))
#     X_train_2[:, i] = np.squeeze(test_scaler.transform(X_train_2[:, i].reshape(len(X_train_2[:, i]), 1)), axis=1)
#     test_scaler_list.append(test_scaler)



# X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_train_2, y_train, shuffle=False, test_size=0.3)
X_train_2, X_valid_2, y_train_2, y_valid_2 = X_train_2, X_train_2, y_train, y_train
print('!!!!!', X_train_2.shape, y_train.shape, X_valid_2.shape, y_valid_2.shape)
train_num = len(X_train_2)
val_num = len(X_valid_2)

# ### visualization
# i_num = 4
# r_num = 100
# s_num = 480
# ch1_sh_df = train_df.iloc[0, i_num:i_num+s_num]
# ch2_sh_df = train_df.iloc[0, i_num+s_num:i_num+s_num+s_num]
# ch3_sh_df = train_df.iloc[0, i_num+s_num+s_num:i_num+s_num+s_num+s_num]
# ch1_sh_np = ch1_sh_df.to_numpy()
# ch2_sh_np = ch2_sh_df.to_numpy()
# ch3_sh_np = ch3_sh_df.to_numpy()
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# xs = np.arange(r_num)
# ys = np.arange(r_num)
# ax.scatter(xs, ys, ch1_sh_np[:r_num], c='red', marker='o', s=15, cmap='Greens')
# ax.scatter(xs, ys, ch2_sh_np[:r_num], c='green', marker='o', s=15, cmap='Greens')
# ax.scatter(xs, ys, ch3_sh_np[:r_num], c='blue', marker='o', s=15, cmap='Greens')
# plt.show()




## 출력 결과 파일 생성
print('출력 결과 파일 생성')
test_df = pd.read_csv('test_full.csv', nrows=end)

X_test_2 = test_df[test_df.columns.difference(['img_name', 'PSNR'])].to_numpy()

# ### test_feature_scaling
# for i in range(X_test_2.shape[1]):
#     test_scaler = test_scaler_list[i]
#     X_test_2[:, i] = np.squeeze(test_scaler.transform(X_test_2[:, i].reshape(len(X_test_2[:, i]), 1)), axis=1)


itr = 1

colsample_bytree = 0.2395378257765287
gamma = 0
learning_rate = 0.013272941370633515
max_depth = 32
n_estimators = 1876
reg_alpha = 0.02152634109282501
reg_lambda = 0.06544801615179391
subsample = 0.7496307600223272
min_child_weight = 1

# colsample_bytree = 1
# gamma = 1.5
# learning_rate = 0.0001
# max_depth = 32
# n_estimators = 100000
# reg_alpha = 2.1
# reg_lambda = 2.3
# subsample = 0.5



def evaluation(np_train_results, np_val_results, my_model_name):
    train_results = np_train_results.copy()
    val_results = np_val_results.copy()

    # train
    train_results = train_results / 1000.0  ##########
    train_results = np.expand_dims(train_results, axis=0)
    train_results = scaler.inverse_transform(train_results)  ##########

    train_sub = pd.DataFrame({'img_name': name_train[:train_num].flatten(),
                              'PSNR': train_results.flatten()})
    train_sub.to_csv('my_train_' + my_model_name + '.csv', index=False)

    # val
    val_results = val_results / 1000.0  ##########
    val_results = np.expand_dims(val_results, axis=0)
    val_results = scaler.inverse_transform(val_results)  ##########

    val_sub = pd.DataFrame({'img_name': name_train[train_num:train_num + val_num].flatten(),
                            'PSNR': val_results.flatten()})
    val_sub.to_csv('my_val_' + my_model_name + '.csv', index=False)

    gt_df = pd.read_csv('train.csv')
    my_train_df = pd.read_csv('my_train_' + my_model_name + '.csv')
    my_val_df = pd.read_csv('my_val_' + my_model_name + '.csv')

    train_mae = MAE(gt_df[:train_num], my_train_df)
    val_mae = MAE(gt_df[train_num:train_num + val_num], my_val_df)

    print(my_model_name, 'Evaluation', 'train_mae, val_mae', train_mae, val_mae)


import time



########## xgboost
# XG = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01, gamma=0, colsample_bytree=1,
#                 max_depth=16, n_estimators=1000).fit(X_train_2, y_train,
#                                                      eval_set=[(X_valid_2, y_valid)], eval_metric='mae')

##########################################

# # 0 TEST
# start = time.time()
# test_model = KernelRidge(alpha=0.6,kernel='polynomial',degree = 2,coef0=2.5)
# test_model.fit(X_train_2, y_train_2)
# # joblib.dump(test_model, 'TEST_model.pkl')
# # test_model = joblib.load('TEST_model.pkl')
# results_TEST = test_model.predict(X_train_2).flatten()
# results_TEST_val = test_model.predict(X_valid_2).flatten()
# print('TEST Model end..', time.time() - start)
# evaluation(results_TEST, results_TEST_val, 'TEST')

# 1
start = time.time()
if is_train:
    xg = xgb.XGBRegressor(colsample_bytree=colsample_bytree, gamma=gamma,
                                 learning_rate=learning_rate, max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                 subsample=subsample, silent=1, min_child_weight=min_child_weight,
                                 random_state=7, nthread=-1)
    xg.fit(X_train_2, y_train_2, eval_set=[(X_valid_2, y_valid_2)], eval_metric='mae', early_stopping_rounds=100)
    joblib.dump(xg, 'xg_model.pkl')
    # joblib.dump(xg, 'xg_scale_model.pkl')

xg = joblib.load('xg_model.pkl')
# xg = joblib.load('xg_scale_model.pkl')
results_xgb = xg.predict(X_train_2).flatten()
results_xgb_val = xg.predict(X_valid_2).flatten()
print('XGBReressor end..', time.time() - start)


# ### cross validation test
# # kfold_xg = xgb.XGBRegressor(colsample_bytree=colsample_bytree, gamma=gamma,
# #                              learning_rate=learning_rate, max_depth=max_depth,
# #                              n_estimators=n_estimators,
# #                              reg_alpha=reg_alpha, reg_lambda=reg_lambda,
# #                              subsample=subsample, silent=1, min_child_weight=min_child_weight,
# #                              random_state=7, nthread=-1)
# #
# # kfold = KFold(n_splits=5)
# # n_iter = 0
# # for train_idx, val_idx in kfold.split(X_train_2):
# #     kfold_train_X, kfold_val_X = X_train_2[train_idx], X_train_2[val_idx]
# #     kfold_train_y, kfold_val_y = y_train_2[train_idx], y_train_2[val_idx]
# #
# #     kfold_xg.fit(kfold_train_X, kfold_train_y, eval_set=[(kfold_val_X, kfold_val_y)], eval_metric='mae', early_stopping_rounds=100)
# #     print(n_iter, 'end..')
# #     n_iter += 1
# # joblib.dump(kfold_xg, 'kfold_xg_model.pkl')
# kfold_xg = joblib.load('kfold_xg_model.pkl')
# results_kfold_xg = kfold_xg.predict(X_train_2).flatten()
# results_kfold_xg_val = kfold_xg.predict(X_valid_2).flatten()
#
# evaluation(results_kfold_xg, results_kfold_xg_val, 'kfold_xg')

# 2
start = time.time()
if is_train:
    gbr = GradientBoostingRegressor(n_estimators=1800,
                                    learning_rate=0.01636280613755809,
                                    max_depth=32,
                                    max_features='sqrt',
                                    min_samples_leaf=5,
                                    min_samples_split=9,
                                    loss='huber',
                                    random_state=1,
                                    validation_fraction=0.3,
                                    n_iter_no_change=100)

    # gbr = GradientBoostingRegressor()
    gbr.fit(X_train_2, y_train_2)
    joblib.dump(gbr, 'gbr_model.pkl')
    # joblib.dump(gbr, 'gbr_scale_model.pkl')

gbr = joblib.load('gbr_model.pkl')
# gbr = joblib.load('gbr_scale_model.pkl')
results_gbr = gbr.predict(X_train_2).flatten()
results_gbr_val = gbr.predict(X_valid_2).flatten()
print('GBReressor end..', time.time() - start)

# 3
start = time.time()
if is_train:
    rfg = RandomForestRegressor()
    rfg.fit(X_train_2, y_train_2)
    joblib.dump(rfg, 'rfg_model.pkl')
    # joblib.dump(rfg, 'rfg_scale_model.pkl')

rfg = joblib.load('rfg_model.pkl')
# rfg = joblib.load('rfg_scale_model.pkl')
results_rfg = rfg.predict(X_train_2).flatten()
results_rfg_val = rfg.predict(X_valid_2).flatten()
print('RandomForest end..', time.time() - start)

# 4
start = time.time()
if is_train:
    etr = ExtraTreesRegressor()
    etr.fit(X_train_2, y_train_2)
    joblib.dump(etr, 'etr_model.pkl')
    # joblib.dump(etr, 'etr_scale_model.pkl')

etr = joblib.load('etr_model.pkl')
# etr = joblib.load('etr_scale_model.pkl')
results_etr = etr.predict(X_train_2).flatten()
results_etr_val = etr.predict(X_valid_2).flatten()
print('ExtraTrees end..', time.time() - start)

# 5
start = time.time()
if is_train:
    hgrb = HistGradientBoostingRegressor()
    hgrb.fit(X_train_2, y_train_2)
    joblib.dump(hgrb, 'hgrb_model.pkl')
    # joblib.dump(hgrb, 'hgrb_scale_model.pkl')

hgrb = joblib.load('hgrb_model.pkl')
# hgrb = joblib.load('hgrb_scale_model.pkl')
results_hgrb = hgrb.predict(X_train_2).flatten()
results_hgrb_val = hgrb.predict(X_valid_2).flatten()
print('HistGradientBoosting end..', time.time() - start)

# 6
start = time.time()
if is_train:
    br = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=20, verbose=1)
    br.fit(X_train_2, y_train_2)
    joblib.dump(br, 'br_model.pkl')
    # joblib.dump(br, 'br_scale_model.pkl')

br = joblib.load('br_model.pkl')
# br = joblib.load('br_scale_model.pkl')
results_br = br.predict(X_train_2).flatten()
results_br_val = br.predict(X_valid_2).flatten()
print('Bagging end..', time.time() - start)


# 7
start = time.time()
if is_train:
    lgb = LGBMRegressor()
    lgb.fit(X_train_2, y_train_2)
    joblib.dump(lgb, 'lgb_model.pkl')
    # joblib.dump(lgb, 'lgb_scale_model.pkl')

lgb = joblib.load('lgb_model.pkl')
# lgb = joblib.load('lgb_scale_model.pkl')
results_lgb = lgb.predict(X_train_2).flatten()
results_lgb_val = lgb.predict(X_valid_2).flatten()
print('LGBM end..', time.time() - start)


# ## Deep learning
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# input = tf.keras.layers.Input((482,))
#
# x1 = tf.keras.layers.Dense(64, activation='relu')(input)
# x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
# x3 = tf.keras.layers.Dense(64, activation='relu')(x2)
# x4 = tf.keras.layers.Add()([x1, x3])
# x4 = tf.keras.layers.Dropout(0.5)(x4)
# x5 = tf.keras.layers.Dense(32, activation='relu')(x4)
# x6 = tf.keras.layers.Dense(32, activation='relu')(x5)
# x7 = tf.keras.layers.Dense(32, activation='relu')(x6)
# x8 = tf.keras.layers.Add()([x5, x7])
# x9 = tf.keras.layers.Dropout(0.5)(x8)
# output = tf.keras.layers.Dense(1)(x9)
#
# model = tf.keras.models.Model(inputs=input, outputs=output)
#
# epochs = 1000
#
# es = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=30)
# mc = ModelCheckpoint('dnn_model.h5', monitor='val_mae', mode='min', verbose=1, save_best_only=True)
#
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mae', metrics=['mae'])
# model.summary()
#
# tf.keras.utils.plot_model(model,
#                           to_file='dnn_plot.png',
#                           show_shapes=True
#                           )
#
# history = model.fit(X_train_2, y_train_2, epochs=epochs, callbacks=[es, mc],
#                     batch_size=64, validation_data=(X_valid_2, y_valid_2))
#
# loaded_model = load_model('dnn_model.h5')
# results_dnn = loaded_model.predict(X_train_2).flatten()
# results_dnn_val = loaded_model.predict(X_valid_2).flatten()
# evaluation(results_dnn, results_dnn_val, 'dnn')


# # wieghts with softmax
# # a = np.array([1-0.7964034790734555, 1-0.7979494178230259, 1-0.8008287089258564, 1-0.7962658048445865,
# #               1-0.8086748009629378, 1-0.8082310484852818, 1-0.7974138878275361])
# a = np.array([1-4.30585670575833e-13, 1-0.03762603208483228, 1-0.260978843083284, 1-0.2891602327516285,
#               1-0.3045774929850348, 1-0.5491403625769229, 1-0.5452765657914316])
# exp_a = np.exp(a)
# sum_exp_a = np.sum(exp_a)
# y = exp_a / sum_exp_a
# w1, w2, w3, w4, w5, w6, w7 = y[0], y[1], y[2], y[3], y[4], y[5], y[6]

while (True):
    print('models etr xgb gb rfg lgb hgrb br')
    print('weights', w1, w2, w3, w4, w5, w6, w7)
    weights_sum = w1 + w2 + w3 + w4 + w5 + w6 + w7

    results = ((w1 * results_etr) + (w2 * results_xgb) + (w3 * results_gbr) + (w4 * results_rfg) \
              + (w5 * results_lgb) + (w6 * results_hgrb) + (w7 * results_br)) / weights_sum
    results_val = ((w1 * results_etr_val) + (w2 * results_xgb_val) + (w3 * results_gbr_val) + (w4 * results_rfg_val) \
              + (w5 * results_lgb_val) + (w6 * results_hgrb_val) + (w7 * results_br_val)) / weights_sum

    ##### EVALUATION
    evaluation(results_etr, results_etr_val, 'etr')
    evaluation(results_xgb, results_xgb_val, 'xgb')
    evaluation(results_gbr, results_gbr_val, 'gb')
    evaluation(results_rfg, results_rfg_val, 'rfg')
    evaluation(results_lgb, results_lgb_val, 'lgb')
    evaluation(results_hgrb, results_hgrb_val, 'hgrb')
    evaluation(results_br, results_br_val, 'br')

    evaluation(results, results_val, 'ensemble')


    ##### TEST
    outputs_xgb = xg.predict(X_test_2).flatten()
    outputs_gbr = gbr.predict(X_test_2).flatten()
    outputs_rfg = rfg.predict(X_test_2).flatten()
    outputs_etr = etr.predict(X_test_2).flatten()
    outputs_hgrb = hgrb.predict(X_test_2).flatten()
    outputs_br = br.predict(X_test_2).flatten()
    outputs_lgb = lgb.predict(X_test_2).flatten()

    outputs = ((w1 * outputs_etr) + (w2 * outputs_xgb) + (w3 * outputs_gbr) + (w4 * outputs_rfg) \
              + (w5 * outputs_lgb) + (w6 * outputs_hgrb) + (w7 * outputs_br)) / weights_sum

    print('outputs.shape', outputs.shape)
    print('outputs')
    print(outputs)

    # Scaling
    outputs = outputs / 1000.0 ##########
    ## xgboost or ensemble
    outputs = np.expand_dims(outputs, axis=0)
    outputs = scaler.inverse_transform(outputs) ##########

    sample_df = pd.read_csv('sample_submission.csv')
    ## xgboost or ensemble
    sample_df['PSNR'] = outputs.flatten()
    ## not xgboost or ensemble
    # sample_df['PSNR'] = outputs

    sample_df.to_csv("my_submission.csv", index=False)

    ####### setting
    input_param = input('input 7 weights:').split()

    try:
        w1 = float(input_param[0])
        w2 = float(input_param[1])
        w3 = float(input_param[2])
        w4 = float(input_param[3])
        w5 = float(input_param[4])
        w6 = float(input_param[5])
        w7 = float(input_param[6])
    except Exception as e:
        print('입력 오류')
        pass






