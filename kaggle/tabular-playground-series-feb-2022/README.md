# Tabular Playground Series - Feb 2022
## 결과
### 요약정보
- 도전기관: 한양대학교
- 도전자: 김진훈
- 최종스코어: 0.98886
- 제출일자: 2022-03-22
- 총 참여 팀수: 1255
- 순위 및 비울: 16.5%
### 결과화면
![leaderboard](./img/leaderboard.png)
## 사용한 방법 & 알고리즘
- Preprocessing: reduce memory usage of DataFrame and use label encoder for the target
- Model: ExtraTreesClassifier
- Postprocessing: Added bias to the probabilities of the ExtraTreesClassifier
## 코드
[`./tabular-playground-series-feb-2022.ipynb`](./tabular-playground-series-feb-2022.ipynb)
## 참고 자료
- Preprocessing: https://www.kaggle.com/code/bhadaneeraj/reduce-memory-usage-of-dataframe/notebook
- Postprocessing: https://www.kaggle.com/code/ambrosm/tpsfeb22-02-postprocessing-against-the-mutants/notebook#The-surprise
