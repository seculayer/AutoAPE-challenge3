## 주차수요 예측 AI 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 125.0751
* 제출일자 : 2022-01-11
* 총 참여 팀 수 : 1630
* 순위 및 비율 :  462(28%)

### 리더보드

![결과](screenshot/scoreCar.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘
* Random Forest
  * ensemble (지도학습) 머신러닝 모델: 여러 개의 decision tree를 결합하는 것이 더 좋은 결과, 성능을 낸다는 아이디어에서 착안
    <img src="screenshot/RandomForest.png" alt="model" style="zoom: 67%;" />
    - Bagging (Bootstrap Aggregating): 여러 개의 트리를 생성하는데, 각 트리 생성 시 training set에 대하여 임의로 n개의 데이터를 선택. 이때 데이터 중복 허용(with replacement)
    ![결과](screenshot/RandomForest2.png)
    - Bagging Features: Feature 선택 시 feature의 부분집합 활용. 일반적으로 M개의 feature가 있다면, 루트 M개의 feature를 선택. 이후 information gain이 높은 feature 선택
    - Classify: 여러 트리 형성 후 도출된 결과 -> 빈도수가 가장 높은 예측값을 최종 결론으로 선택.<br>
      ex. 8개의 트리를 형성하고 나온 예측값이 5개가 very good이라면, 예측값은 'very good'으로 분류
  <br><br>
 
 2. 문제 해결 방법
 * data 전처리: 
   * 결측치 처리
   * 컬럼명 변경
   * 지역명 숫자로 mapping 
   * '전용면적' 데이터에 대한 data pre-processing
 * modeling (모델 정의 및 모델 학습) -> RandomForestRegressor으로 모델 정의 및 train data에 대하여 모델 학습
 * test data 예측 -> RandomForest Regressor으로 test data predict

-----------

### 코드

['./주차수요 예측 AI 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EC%A3%BC%EC%B0%A8%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EC%A3%BC%EC%B0%A8%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[RandomForest](https://medium.com/greyatom/a-trip-to-random-forest-5c30d8250d6a)
