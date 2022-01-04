## 구내식당 식수 인원 예측 AI 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 119.32466
* 제출일자 : 2022-01-03
* 총 참여 팀 수 : 1573
* 순위 및 비율 :  496(31%)

### 리더보드

![결과](screenshot/scoreEat.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘 및 주요 개념
* KFold
  * train / validation set에 대하여 k개의 폴드 세트에 K번의 학습(train) 및 검증(validation)을 진행
  * train을 1회만 실시할 경우 과적합(overfit)할 가능성이 크기 때문에 교차 검증 실시
  * 한계: 레이블 값의 분포를 반영하지 못함 (데이터의 불균형성)<br>
    ex. Label A와 LabelB의 개수 비가 100:1 -> 불균형
 * StratifiedKFold
   * KFold의 한계(레이블 값의 분포 반영 못함)을 해결
   * 데이터 분포에 따라 train / validation dataset split
* RandomForest Classifier
* log_loss (cross-entropy loss): 손실함수
  * 모델이 label 값을 예측할 확률을 이용하여 평가
  * log_loss가 낮을수록 좋은 성능을 보이는 것, 즉 예측을 잘한 것
  <br><br>
 
 2. 문제 해결 방법
 * data 전처리: column '요일'을 숫자로 mapping -> categorical data만 feature로 선택
 * modeling (모델 정의 및 모델 학습) -> RandomForestRegressor으로 모델 정의 및 train data에 대하여 모델 학습
 * test data 예측 -> RandomForest Regressor으로 test data predict

-----------

### 코드

['./신용카드 사용자 연체 예측 AI 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EA%B5%AC%EB%82%B4%EC%8B%9D%EB%8B%B9%20%EC%8B%9D%EC%88%98%20%EC%9D%B8%EC%9B%90%20%EC%98%88%EC%B8%A1%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EA%B5%AC%EB%82%B4%EC%8B%9D%EB%8B%B9%20%EC%8B%9D%EC%88%98%20%EC%9D%B8%EC%9B%90%20%EC%98%88%EC%B8%A1%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
[StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
[Log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
