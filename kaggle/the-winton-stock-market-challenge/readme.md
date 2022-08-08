### 요약정보 
- 도전기관 : 한양대 
- 도전자 : 김현주 
- 최종스코어 :  1769.89707
- 제출일자 : 2022-08-07
- 총 참여 팀수 : 829
- 순위 및 비율 : 8(0.9%)

### 결과화면 
![result](./img/first_score.PNG) 
![result](./img/first_score2.PNG) 

### 사용한 방법 & 알고리즘 
- XGBoost 알고리즘을 사용하여 모델링함 
- 기존의 square loss function 대신 liear loss function을 사용
- hyper parameter 튜닝을 통한 성능향상
- XGBRegressor( objective = 'reg:pseudohubererror',
                          tree_method = 'gpu_hist',
                          max_depth = 5,
                          min_child_weight = 3,
                          gamma = 0,
                          subsample = 0.9,
                          colsample_bytree = 0.9,
                          alpha = 0,
                          learning_rate = 0.01,
                          n_estimators = 700)