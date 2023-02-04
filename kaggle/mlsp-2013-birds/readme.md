### 요약정보 
- 도전기관 : 한양대 
- 도전자 : 이자윤 
- 최종스코어 :  0.91973
- 제출일자 : 2023-02-04
- 총 참여 팀수 : 79
- 순위 및 비율 : 14(17.72%)

### 결과화면 
![result](./img/1.PNG) 
![result](./img/2.PNG) 

### 사용한 방법 & 알고리즘 
- Features: 
  CNN multi-label classification result (Densenet121)

  Segment K-Means clustering

  Recording location

  Recording area (clustered nearby locations into 6 areas)

- MultiOutputClassifier and RandomForestClassifier are used for model training

### 코드

[./MLSP_bird.ipynb](./MLSP_bird.ipynb)

### 참고자료

- [https://www.kaggle.com/competitions/mlsp-2013-birds/discussion/5457#29217](https://www.kaggle.com/competitions/mlsp-2013-birds/discussion/5457#29217)
- [https://github.com/lidan1/multidensenet](https://github.com/lidan1/multidensenet)
