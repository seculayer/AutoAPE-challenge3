## 물류 유통량 예측 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 5.70357
* 제출일자 : 2022-01-12
* 총 참여 팀 수 : 520
* 순위 및 비율 :  150(29%)

### 리더보드

![결과](screenshot/scoreGoods.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘 및 주요개념
* CatBoost (Categorical Boosting)
  * parameter tuning을 하지 않아도 좋은 성능을 낸다
    - default parameter로 좋은 결과를 도출하기에, parameter tuning에 드는 시간 감소
  * categorical feature를 다룸
    - numerical feature가 아닌, non-numerical feature에 대해 잘 작동함
    - categorical feature를 labelEncode 하는 등의 시간 감소 
  * 높은 정확도
    - 과적합(overfit) 감소
  <img src="screenshot/catboostGood.png" alt="model" style="zoom: 67%;" />
  
 
 2. 문제 해결 방법
 * modeling (모델 정의 및 모델 학습)
   - catboost Regressor로 모델 정의 및 train data에 대하여 모델 학습
   - catboost를 활용하기 때문에 LabelEncoder를 활용하여 categorical data를 numerical data로 변환할 필요가 없다
 * test data 예측 -> catboost Regressor으로 test data predict

-----------

### 코드

['./물류 유통량 예측 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[catboost](https://catboost.ai/) 
