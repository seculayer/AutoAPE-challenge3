# Africa Soil Property Prediction Challenge
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김진훈
- 최종스코어: 0.39259
- 제출일자: 2022-11-16
- 총 참여 팀수: 1602
- 순위 및 비울: 28.4%
### 결과화면
![leaderboard](./img/leaderboard.png)
## 사용한 방법 & 알고리즘
- Preprocessing: Use label encoder for the categorical features and the memory usage reducer function from https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook
- Model implemented by tinrtgu from https://www.kaggle.com/competitions/avazu-ctr-prediction/submissions
- Ensemble of gradient boosting models: catboost, lgbm, xgboost
- Ensemble of the ensemble of gradient boostin models and the model implemented by tinrtgu
- How to use the code:
   - Download the dataset from kaggle
   - Run mem_reducer.ipynb
   - Run avazu-ctr-prediction.ipynb
## 코드
[`./avazu-ctr-prediction.ipynb`](./avazu-ctr-prediction.ipynb)
## 참고 자료
- Memory usage reducing function: https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook
- tinrtgu's model: https://www.kaggle.com/competitions/avazu-ctr-prediction/submissions