### 요약정보 
- 도전기관 : 한양대 
- 도전자 : 이자윤 
- 최종스코어 :  0.30570
- 제출일자 : 2022-11-24
- 총 참여 팀수 : 1241
- 순위 및 비율 : 205(16.52%)

### 결과화면 
![result](./img/1.PNG) 
![result](./img/2.PNG) 

### 사용한 방법 & 알고리즘 
- Features: 
  Number of: words, unique words, characters, stopwords, punctuations, upper words, title words, most words

  Mean word length

  Fraction of: noun, adj, verb

  Glove vectors

  TfidfVectorizer vectors

  CountVectorizer vectors

  Word frequency using NLTK

- VotingClassifier is used to create 11 sets of vectorizer features 

- SVD is used to create new sets of vectorizer features

- XGBoost is used for model training

### 코드

[./spooky.ipynb](./spooky.ipynb)

### 참고자료

- [面向文本分類的特徵工程——kaggle文本分類比賽](https://www.twblogs.net/a/5b8d73a82b717718833e1640)

- [Beginner's Tutorial: Python](https://www.kaggle.com/code/rtatman/beginner-s-tutorial-python)





