# Springleaf Marketing Response
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김진훈
- 최종스코어: 0.79696
- 제출일자: 2022-01-07
- 총 참여 팀수: 2221
- 순위 및 비울: 2.6%
### 결과화면
![leaderboard](./img/leaderboard.png)
## 사용한 방법 & 알고리즘
- Preprocessing (followed the approach of Abishek Thatur)
  - Eliminated features with more than 2000 missing values
  - Replaced the rest of missing values with -1
  - Label-encoded the categorical features
- Model: XGBClassifier
## 코드
[`./springleaf-marketing-response.ipynb`](./springleaf-marketing-response.ipynb)
## 참고 자료
- Preprocessing and model parameters from https://www.kaggle.com/c/springleaf-marketing-response/discussion/17089