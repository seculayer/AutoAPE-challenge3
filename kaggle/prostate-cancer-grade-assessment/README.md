# Prostate cANcer graDe Assessment (PANDA) Challenge

## 결과

### 요약정보

- 도전기관 : 한얀대학교
- 도전자 : 왕전
- 최종스코어 : 0.88968
- 제출일자 : 2022-06-07
- 총 참여 팀 수 : 1010
- 순위 및 비율 : 131(12.97%)

### 결과화면

![](./img/score.PNG)

## 사용한 방법 & 알고리즘

efficientNet을 사용합니다.efficientNet의 구조는 from efficientnet_pytorchimport EfficientNet을 통해 B0 모델을 출력할 수 있다.
EfficientNet은 네트워크 깊이, 폭 및 해상도를 통합 스케일링하기 위해 compound scaling 방법을 사용합니다.
B0 네트워크를 기반으로 block 블록의 매개변수는 width_coefficient 및 depth_coefficient에 따라 조정되며, 여기서 width_coefficient는 filters의 크기, 
즉 네트워크의 channel을 결정하고 depth_coefficient는 num_repeat의 크기, 즉 네트워크의 깊이를 결정합니다.


## 코드

[](./result.ipynb)







