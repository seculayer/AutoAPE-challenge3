## KNOW기반 직업 추천 알고리즘 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 0.37487
* 제출일자 : 2022-01-05
* 총 참여 팀 수 : 905
* 순위 및 비율 :  141(15.58%)

### 리더보드

![결과](screenshot/screenKnow.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘
* ensemble (지도학습) 머신러닝 모델: 여러 개의 decision tree를 결합하는 것이 더 좋은 결과, 성능을 낸다는 아이디어에서 착안
  <img src="screenshot/ensemble.png" alt="model" style="zoom: 67%;" />
  * Classify: 여러 트리 형성 후 도출된 결과 -> 빈도수가 가장 높은 예측값을 최종 결론으로 선택.<br>
    <br>ex. 8개의 트리를 형성하고 나온 예측값이 5개가 very good이라면, 예측값은 'very good'으로 분류 
  <br><br>
 
 2. 문제 해결 방법
 * data 전처리
   * LabelEncoding
 * modeling (모델 정의 및 모델 학습): HardVoting Classifier
   * Logistic Regression
   * RandomForest Classifier
   * DecisionTree Classifier
 * test data 예측 -> model로 test data predict

-----------

### 코드

['./KNOW기반 직업 추천 알고리즘 경진대회.ipynb](https://github.com/gjdls01/seculayer_challenge3/blob/main/dacon/KNOW%EA%B8%B0%EB%B0%98%20%EC%A7%81%EC%97%85%20%EC%B6%94%EC%B2%9C%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/KNOW%EA%B8%B0%EB%B0%98%20%EC%A7%81%EC%97%85%20%EC%B6%94%EC%B2%9C%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[Voting Classifier](https://www.codingninjas.com/codestudio/library/the-voting-classifier)

