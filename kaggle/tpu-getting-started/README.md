#Petals to the Metal - Flower Classification on TPU
##결과
###요약정보
-도전기관 :한양대학교
-도전자: 탄텐보
-최중스코어:0.94962
-저출일자:2022.3.13
-중 참여 팀 수 :143
-순위 및 비율: 36(25.17%)
###결과화면 
![65e6eb3f9e75a212206afbc71fbea31](./img/65e6eb3f9e75a212206afbc71fbea31.png)

##사용한 방법&알고리즘 
신경망 모델을 만들 때 분포 전략을 사용한다. 
TensorFlow는 각 코어에 대해 하나씩 8개의 서로 다른 모델 복제본을 생성하여 8개의 TPU 코어에 교육을 배포한다.
-Full connnected neural network
-20nodes
-9nodes
-1output
##코드
[](./test6.ipynb)
##참고자료
https://wikidocs.net/119990
