# HuBMAP - Hacking the Kidney

## 결과

### 요약정보

- 도전기관 : 한얀대학교
- 도전자 : 왕전
- 최종스코어 : 0.9360
- 제출일자 : 2022-09-26
- 총 참여 팀 수 : 1200
- 순위 및 비율 : 285(23.75%)

### 결과화면

![](./img/score.PNG)

## 사용한 방법 & 알고리즘

모델: U-Net SeResNext101 + CBAM + Hypercolumns + deepsupervision
손실 함수: deep supervision with bcelloss + lovasz-hingeloss + classification bce loss
외부 데이터 세트 & 의사 라벨: 공개 테스트 세트 + dataset_a_dib
검증 세트 분할: 수동 지정, 샘플링
예측 trick : 예측 결과의 중간 위치 결과를 사용하여 edge effect 제거


## 코드

[](./notebook.ipynb)







