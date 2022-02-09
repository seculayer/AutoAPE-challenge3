# 소설 작가 분류 AI 경진대회

## 결과

### 요약정보

- 도전기관 : 시큐레이어
- 도전자 : 김준혁
- 최종스코어 : 0.29277 
- 제출일자 : 2021-10-22
- 총 참여 팀 수 : 356
- 순위 및 비율 : 59(16.57%)

### 결과화면
![novel_learderboard](./img/novel_learderboard.PNG)


## 사용한 방법 & 알고리즘

- DNN 모델과 CNN 모델을 만들어 학습 후 앙상블
- 텍스트 전처리
  - DNN은 SentencePieceTrainer를 이용하여 전처리 하였다.
  - CNN은 Bert를 이용하여 전처리를 하였다.
- DNN과 CNN모델 3개를 사용하였다.
- K-Fold를 사용하여 성능이 제일 좋은 모델을 앙상블에 사용하였다.

## 코드
['./Novel_Author_Classification_AI_Contest.ipynb'](./Novel_Author_Classification_AI_Contest.ipynb)


## 참고 자료
