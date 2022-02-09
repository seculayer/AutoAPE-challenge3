## 영화 리뷰 감성분석 경진대회

------------

### 결과

----------------

### 결과 요약

* 도전기관 : 시큐레이어
* 도전자 : 허인
* 최종스코어 : 0.8635
* 제출일자 : 2022-01-19
* 총 참여 팀 수 : 574
* 순위 및 비율 :  38 (7%)

### 리더보드

![결과](screenshot/scoreMovie.png)

----------

### 알고리즘 & 문제 해결 방법

1. 알고리즘
* KcELECTRA (Korean comments ELECTRA)
  * 자연어 처리 분야에서 좋은 성능을 보여주는 모델 (한국어 성능 한계 개선) - 한국어 댓글을 이용하여 학습한 ELECTRA 모델
  * 한국어 Transformer 계열 모델과의 차이
    - Transformer 계열 모델: 한국어 위키, 뉴스 기사, 책 등 정제된 데이터를 기반으로 학습한 모델
    - KcELECTRA: 네이버 뉴스에서 댓글, 대댓글을 수집하여 학습한 모델 (정제되지 않은 데이터, 구어체 특징의 신조어, 오탈자 등 공식적 글쓰기에서 나타나지 않는 표현)
  * KcBERT 대비 데이터셋 증가, vocab 확장
    ![결과](screenshot/electra2.png) 
  * ELECTRA
    - 학습의 효율성을 위해 RTD 사용<br>
      - RTD: 일부 토큰을 Generator에서 얻은 가짜 토큰으로 치환 - Discriminator로 들어온 토큰이 실제 토큰인지, Generator을 통해 생성된 가짜 토큰인지 맞힘<br>
      - 모든 토큰에 대해 이진 분류를 해야 하기에 모든 토큰에 대하여 loss를 계산해야 함 (모든 토큰에 대해 학습하기에 효과적)<br>
        ![결과](screenshot/electra.png) 
  <br><br><br>
 
 2. 문제 해결 방법
 * argument 설정
 * Pytorch Lightning 활용해서 모델 만들기
 * 학습
 * 예측
 ![결과](screenshot/resultMovie.png)
 
-----------

### 코드

['./영화 리뷰 감성분석 경진대회.ipynb](https://github.com/gjdls01/AutoAPE-challenge3/blob/main/dacon/%EC%98%81%ED%99%94%20%EB%A6%AC%EB%B7%B0%20%EA%B0%90%EC%84%B1%EB%B6%84%EC%84%9D%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C/%EC%98%81%ED%99%94%20%EB%A6%AC%EB%B7%B0%20%EA%B0%90%EC%84%B1%EB%B6%84%EC%84%9D%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C.ipynb)

-----------

### 참고자료

[KcELECTRA Github](https://link.ainize.ai/3ezh2SS)
