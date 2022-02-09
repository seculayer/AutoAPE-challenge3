## 뉴스 토픽 분류 AI 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 0.78028
* 제출일자 : 2022-01-10
* 총 참여 팀 수 : 828
* 순위 및 비율 :  245(29%)

### 리더보드

![결과](screenshot/scoreNews.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘
* LightGBM (Light Gradient descent Boosting Machine)
  1) 용어
     - Light -> 빠른 속도 
     - GBM -> 잔여오차(residual error)
  2) 간단한 설명
     - Tree 기반 학습 알고리즘 (Gradient Boosting Framework)
     - 수직 확장(leaf-wise). 반대는 수평 확장이라고 칭한다(level-wise)
     - 과적합에 민감하며, 작은 크기의 데이터는 과적합될 확률이 크기에 대용량의 데이터에 활용
  3) parameter (주요 parameter 및 그외 파라미터)
     - objective
       * regression -> 회귀
       * binary, multiclass -> 분류
     - metric: MAE, RMSE, binary_logloss, AUC, cross_entropy etc
     - learning rate: 일반적으로 0.01 ~ 0.1로 설정
     - n_estimators: 학습횟수. 값이 너무 크면 과적합(overfit)할 가능성이 있다 (default: 100)
     - max_depth : 결정 트리(decision tree)의 max_depth과 같은 개념. 수직 확장하기에 max_depth가 매우 크다 (default값은 -1 -> 0보다 작은 것은 깊이 제한이 없다는 뜻)
     - min_child_samples : 결정트리의 min_samples_leaf와 같은 개념 (default: 20)
     - num_leaves : 하나의 트리가 가질 수 있는 최대 leaf 개수 (default: 31)
     - early_stopping_rounds : 학습 조기종료를 위한 early stopping interval 값
       * early_stopping_rounds = 100, n_estimators = 1000 이라고 가정하자. 학습이 1000회에 도달하지 않더라도 예측 오류가 100번 동안 줄어들지 않으면(validation dataset을 학습시켰을 때 그 성능이 더이상 향상하지 않으면) 중단 -> 시간 감소
     - feature_fraction
       * 값이 1보다 작을 경우, 그 비율만큼 feature를 랜덤하게 추출하여 학습<br>
         ex. feature_fraction = 0.5일 경우, feature의 50%만 랜덤하게 추출 -> 과적합 방지, 속도 향상
     - 그외: min_data_in_leaf / feature_fraction / bagging_fraction / lambda / min_gain_to_split / max_cat_group<br><br>
  <img src="screenshot/lgbm.jpg" alt="model" style="zoom: 40%;" />
  
 
 2. 문제 해결 방법
 * data 전처리
   * target('title')에서 불필요한 정보(조사, 어미, 구두점) 제거 -> 어간 추출
 * train / valid dataset split (80:20)
 * modeling (모델 정의 및 모델 학습) -> LightGBM으로 모델 정의 및 train data에 대하여 모델 학습, 평가
 * test data 예측 -> LightGBM으로 test data predict

-----------

### 코드

['./뉴스 토픽 분류 AI 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EB%89%B4%EC%8A%A4%20%ED%86%A0%ED%94%BD%20%EB%B6%84%EB%A5%98%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EB%89%B4%EC%8A%A4%20%ED%86%A0%ED%94%BD%20%EB%B6%84%EB%A5%98%20AI%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[LightGBM](https://lightgbm.readthedocs.io/en/latest/)
