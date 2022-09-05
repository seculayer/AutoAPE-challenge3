# Predict Future Sales
---
# 결과
---
### 요약정보
* 도전기관 : 시큐레이어
* 도전자 : 박희수
* 최종 스코어 : 0.92075
* 제출 일자 : 202-08-11
* 총 참여 팀 수 : 15357
* 순위 및 비율 : 4710 (30%)

# 결과 화면
![pfs_score](https://ifh.cc/g/YFFBZ3.png)
![pfs_rank](https://ifh.cc/g/Z93gRQ.png)

# 사용한 방법 & 알고리즘
---
* xgb 모델 사용
* 파생 변수 생성
* sales 데이터를 이용한 판매량 예측
* 달 마다 새로운 lag_feature 추가

# 코드

[predict future sales.ipynb](./predict future sales.ipynb)

---
# 참고자료
---
##### https://www.kaggle.com/code/werooring/top-3-5-lightgbm-with-feature-engineering
