## 물류 유통량 예측 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 7.83695
* 제출일자 : 2022-01-03
* 총 참여 팀 수 : 469
* 순위 및 비율 :  241(51%)

### 리더보드

![결과](screenshot/scoreGoods.PNG)

----------

### 알고리즘 & 문제 해결 방법

* LabelEncoder
  * categorical data -> numerical data 
  * 앙상블 학습의 핵심은 여러 개의 약 분류기 (Weak Classifier)를 결합하여 강 분류기(Strong Classifier)를 만드는 것이다. 그리하여 모델의 정확성이 향상된다.

* LightGBM Regressor
  * 머신러닝 Ensemble 방법 중 하나.
  * 우선, 데이터로부터 부트스트랩을 한다. (복원 랜덤 샘플링) 부트스트랩한 데이터로 모델을 학습시킨다. 그리고 학습된 모델의 결과를 집계하여 최종 결과 값을 구한다.
  * Categorical Data는 투표 방식(Votinig)으로 결과를 집계하며, Continuous Data는 평균으로 집계한다.
  * Categorical Data일 때, 투표 방식으로 한다는 것은 전체 모델에서 예측한 값 중 가장 많은 값을 최종 예측값으로 선정한다. 6개의 결정 트리 모델이 있다고 하자. 4개는 A로 예측했고, 2개는 B로 예측했다면 투표에 의해 4개의 모델이 선택한 A를 최종 결과로 예측한다. 
  * 평균으로 집계한다는 것은 말 그대로 각각의 결정 트리 모델이 예측한 값에 평균을 취해 최종 Bagging Model의 예측값을 결정한다.

<img src="screanshot/model.png" alt="model" style="zoom: 67%;" />

-----------

### 코드

['./물류 유통량 예측 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EB%AC%BC%EB%A5%98%20%EC%9C%A0%ED%86%B5%EB%9F%89%20%EC%98%88%EC%B8%A1%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
