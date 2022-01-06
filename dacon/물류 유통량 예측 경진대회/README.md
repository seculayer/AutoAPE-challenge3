## 물류 유통량 예측 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 5.70357
* 제출일자 : 2022-01-06
* 총 참여 팀 수 : 488
* 순위 및 비율 :  150(31%)

### 리더보드

![결과](screenshot/scoreGoods.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘 및 주요개념
* CatBoost
  * Decision Tree(결정트리)와 GBM(Gradient Boosting Machine)에 기반하여 만들어진 알고리즘
  * GBM의 문제(과적합, overfitting) 해결
  * categorical data에 대하여 잘 작동
  <img src="screenshot/LGBM.jpg" alt="model" style="zoom: 67%;" />
  
 
 2. 문제 해결 방법
 * data load
 * data 전처리 -> LabelEncoder 활용하여 categorical data인 column'물품_카테고리'를 numerical data로 변환
 * modeling (모델 정의 및 모델 학습) -> catboost Regressor로 모델 정의 및 train data에 대하여 모델 학습
 * test data 예측 -> catboost Regressor으로 test data predict

-----------

### 코드

['./물류 유통량 예측 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 
