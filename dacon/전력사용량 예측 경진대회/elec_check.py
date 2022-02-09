import numpy as np 
import pandas as pd 
import lightgbm
from lightgbm import LGBMRegressor 
import catboost
from catboost import CatBoostRegressor
import sklearn
from sklearn.model_selection import KFold
import IPython
from IPython.display import clear_output
import copy


FILE_PATH = './energy/'

# CATBOOST 평가 파라미터
cat_mae_params = {
    'objective': 'MAE',
    'n_estimators': 10000,
    'early_stopping_rounds': 4, 
} #catboost hyper parameter

# LGBM 평가 파라미터
lgbm_mae_params = {
    'objective': 'MAE',
    'boosting_type': 'goss',
    'n_estimators': 10000,
    'early_stopping_round': 15, 
    'num_leaves':39,
} #lightgbm hyper parameter

#cooling degree hour를 구현
def CDH(xs): 
    ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(xs[:(i+1)]-26))
        else:
            ys.append(np.sum(xs[(i-11):(i+1)]-26))
    return np.array(ys) 

#iqr 이상치제거
#iqr(Q3 - Q1): 사분위수의 상위 75% 지점의 값과 하위 25% 지점의 값 차이
def detect_outliers(df,ratio):  
    outlier_indices = [] 
    Q1 = np.percentile(df, 25) 
    Q3 = np.percentile(df, 75) 
    IQR = Q3 - Q1 
    outlier_step = ratio * IQR 
    return ~(df < Q1 - outlier_step) | (df > Q3 + outlier_step)

train_df = pd.read_csv(FILE_PATH+'train.csv', encoding = "cp949") #train_csv

# 시간 특징을 가지는 피처를 추가
train_df['date_time'] = pd.to_datetime(train_df['date_time'])
train_df['dayofyear'] = train_df['date_time'].dt.dayofyear
train_df['hour'] = train_df['date_time'].dt.hour
train_df['weekday'] = train_df['date_time'].dt.weekday 

# 시간 인코딩
train_df['hour_te'] = np.sin(2*np.pi*(train_df['hour'])/23) # 원래 24가 정상인데 23이 더 좋은 결과를 가져온다;  # 원본 24
train_df['hour_te1'] = np.cos(2*np.pi*(train_df['hour'])/23) 

t = 9/5*train_df['기온(°C)']
train_df['불쾌지수'] = t - 0.55*(1-train_df['습도(%)']/100)*(t-26)+32
train_df['불쾌지수'] = pd.cut(train_df['불쾌지수'], bins = [0, 68, 75, 80, 200], labels = [1,2,3,4]) #불쾌지수는 카테고리로 나누는게 성능상승에 도움이 됨                                 

train_dfs  = []
for i in range(1,61):
    train_dfs.append(train_df[train_df['num']==i]) 
    
    
for i in range(len(train_dfs)):
    train_dfs[i] = train_dfs[i].drop(columns=['풍속(m/s)','강수량(mm)','일조(hr)','num',
                                              'date_time','비전기냉방설비운영','태양광보유']) # 쓸모없는 특징 drop 
    

test_df = pd.read_csv(FILE_PATH+'test.csv', encoding = "cp949")
test_df['date_time'] = pd.to_datetime(test_df['date_time'])

for i in range(1,61):
    test_df[test_df['num']==i] = test_df[test_df['num']==i].interpolate(method='values') #기상예보값 interpolate 
    
test_df['dayofyear'] = test_df['date_time'].dt.dayofyear
test_df['hour'] = test_df['date_time'].dt.hour
test_df['weekday'] = test_df['date_time'].dt.weekday #time feature

test_df['hour_te'] = np.sin(2*np.pi*(test_df['hour'])/23)
test_df['hour_te1'] = np.cos(2*np.pi*(test_df['hour'])/23) #time encoding hour

t = 9/5*test_df['기온(°C)']
test_df['불쾌지수'] = t - 0.55*(1-test_df['습도(%)']/100)*(t-26)+32
test_df['불쾌지수'] = pd.cut(test_df['불쾌지수'], bins = [0, 68, 75, 80, 200], labels = [1,2,3,4]) #불쾌지수는 카테고리로 나누는게 성능상승에 도움이 됨
    
test_dfs  = []
for i in range(1,61):
    test_dfs.append(test_df[test_df['num']==i])
    
for i in range(len(test_dfs)):
    test_dfs[i] = test_dfs[i].drop(columns=['풍속(m/s)','강수량(mm, 6시간)','일조(hr, 3시간)','num',
                                            'date_time','비전기냉방설비운영','태양광보유']) #쓸모없는 특징 drop

    
for i in range(60): #cdh 특징 추가
    train_dfs[i]['cdh'] = CDH(np.concatenate([train_dfs[i]['기온(°C)'].values,test_dfs[i]['기온(°C)'].values]))[:-len(test_dfs[i])]
    test_dfs[i]['cdh'] = CDH(np.concatenate([train_dfs[i]['기온(°C)'].values,test_dfs[i]['기온(°C)'].values]))[-len(test_dfs[i]):]

    
#train_x와 train_y로 나눔    
train_x = [] 
train_y = []
for i in range(len(train_dfs)):
    train_x.append(copy.deepcopy(train_dfs[i][train_dfs[i].columns[1:]])) 
    train_y.append(copy.deepcopy(train_dfs[i][train_dfs[i].columns[0]]))

#이상치 제거 iqr은 1.25
for i in range(60):    
    idx = detect_outliers(train_y[i],1.25)
    train_y[i] = train_y[i][idx]
    train_x[i] = train_x[i][idx]


#과적합 방지를 위해 여러 k_fold로 반복하도록 설정
#특징중 몇개를 뺄경우 성능 향상을 기대할수 있고 과적합 또한 방지 가능하다
random_seed = 0
dcs = [[],['기온(°C)'], ['습도(%)'], ['hour_te','hour_te1'], ['불쾌지수'], ['cdh']]
ks = [2,3,4,5,6,7,8,9,10,4]

print(train_dfs[0])


answer_df = pd.read_csv(FILE_PATH+'sample_submission.csv', encoding = "cp949")

for dc in dcs:#d특정 feature dc를 drop 시킴
    for k in ks:#kfold 의 nspilt 의 값 k
        folds = []
        for i in range(len(train_dfs)):
            cross=KFold(n_splits=k,shuffle=True,random_state=random_seed)
            fold=[]
            for train_idx, valid_idx in cross.split(train_x[i], train_y[i]):
                fold.append((train_idx, valid_idx))
            folds.append(fold)
            
        for i in range(len(train_dfs)):
            for fold in range(k):
                print(dc,random_seed,k,i)
                train_idx, valid_idx = folds[i][fold]
                X_train=np.array(train_x[i].drop(columns=dc).iloc[train_idx])
                y_train=np.array(train_y[i].iloc[train_idx])
                X_valid=np.array(train_x[i].drop(columns=dc).iloc[valid_idx])
                y_valid=np.array(train_y[i].iloc[valid_idx])

                #catboost 학습 
                model=CatBoostRegressor(**cat_mae_params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
                v = model.predict(np.array(test_dfs[i][train_x[i].drop(columns=dc).columns])) * 0.5     # 원본 0.3
                
                #lgbm 학습 
                model=LGBMRegressor(**lgbm_mae_params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)        
                v += model.predict(np.array(test_dfs[i][train_x[i].drop(columns=dc).columns])) * 0.5    # 원본 0.7
                
                # 위 모델 두개 앙상블(모델 가중치 합산)
                answer_df['answer'].iloc[(i)*168 : (i+1)*168] += v/(len(ks)*k*len(dcs))
                clear_output(True) 
                
        random_seed += 1
answer_df.to_csv(FILE_PATH+'energy_output_1101.csv', index=False) #파일 저장