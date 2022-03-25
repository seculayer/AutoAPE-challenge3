# **What's Cooking? (Kernels Only)**

## 결과

### 요약정보

- 도전기관 : 한양대학교
- 도전자 : 왕전
- 최종스코어 :  0.723
- 제출일자 : 2021-03-18
- 총 참여 팀 수 : 2426
- 순위 및 비율 : 398(16.41%)

### 결과화면

![](./img/1.PNG)

## 사용한 방법 & 알고리즘

Additive angular margin loss function을 이용하여 높은 분별력을 얻는다.

Arcface training 모델을 사용하여 그림과 텍스트를 인코딩한 다음 pool한다. 그리고 batch normalization과 feature wise regularization을 사용하여 embedding을 출력한다. 정규화된 embedded와 normalized 가중행렬을 곱하여 코드를 생성한다.

이 경기는 멀티 모드의 경기입니다. 우리는 이미지, 텍스트, 기타 정보를 가지고 있다. 우리는 이 모드들을 잘 사용해야 한다.

그림 삽입과 텍스트 삽입을 표준화한 다음 연결하여 조합 유사도를 계산합니다. 그래서 우리는 다음과 같이 받아들인다. 이미지 임베디드 강력 추천 제품, 텍스트 임베디드 강력 추천 제품, 이미지 및 텍스트 임베디드 적당히 추천된 제품입니다.

## 코드

[](./shoppe.ipynb)

## 참고 자료

