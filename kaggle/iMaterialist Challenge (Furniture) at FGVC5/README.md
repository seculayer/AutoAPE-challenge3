# iMaterialist Challenge (Furniture) at FGVC5

## 결과

### 요약정보

- 도전기관 : 한양대학교
- 도전자 : 탄텐보
- 최종스코어 : 0.96404
- 제출일자 : 2022-07-11
- 총 참여 팀 수 : 426
- 순위 및 비율 : 305(71.60%)

### 결과화면

![c5d788200ed7f92068c6822a5eb6484.png](./img/c5d788200ed7f92068c6822a5eb6484.png)

## 사용한 방법 & 알고리즘

데이터에 이미지 URL이 있습니다. 그리고 다른 호스팅에서는 대상 클래스의 다른 분포가 있을 것이라고 생각하는 것이 이치에 맞습니다. 또는 이미지 파일 이름에서 의미 있는 단어를 찾을 수도 있습니다.

이제 URL에 대한 텍스트 분류 알고리즘을 훈련하고 무엇을 얻을 수 있는지 봅시다.

### DATA
아래에 설명된 모든 데이터는 JSON 형식의 txt 파일입니다.


train.json: 이미지 URL 및 레이블이 있는 학습 데이터

validation.json: train.json과 동일한 형식의 검증 데이터

test.json: 참가자가 예측을 생성해야 하는 이미지. 이미지 URL만 제공됩니다.

sample_submission_randomlabel.csv: 제출 파일 형식을 설명하기 위해 무작위 예측이 포함된 제출 파일 예시

교육 데이터 세트에는 128개 가구 및 가정용품 클래스의 이미지가 포함되며 각 이미지에 대해 하나의 실측 레이블이 있습니다. 총 194,828개의 교육용 이미지와 6,400개의 검증용 이미지, 12,800개의 테스트용 이미지가 포함되어 있습니다.
훈련 및 검증 세트는 아래와 같은 형식을 가집니다.


### Model
훈련 데이터의 모든 URL에서 가장 일반적인 2,500개의 1-, 2- 및 3-gram을 찾았습니다. 더 알고 싶다면 여기 좋은 문서가 있습니다. 이제 URL에서 해당 n-gram의 발생 횟수를 계산하고 tf-idf라는 평활화를 적용할 수 있습니다. 이제 이러한 기능에 대해 로지스틱 회귀를 훈련해 보겠습니다. 문제는 우리가 많은 기능과 많은 포인트를 가지고 있기 때문에 LR은 (이 노트북 환경에서) 영원히 걸릴 것입니다. 따라서 우리는 모델을 훈련시키기 위해 수천 개의 임의 포인트를 사용할 것입니다.

## 코드
[this-is-a-text-classification-competition.ipynb](./this-is-a-text-classification-competition.ipynb)

## 참고 자료