# Python Script 사용하기

## 의존성

- Python
- [Poetry](https://python-poetry.org/): Python 패키지 관리자

### Poetry

Poetry를 설치했다면, 프로젝트 경로에서 `poetry install`을 실행하여 의존성을 설치합니다.

## Metadata를 CSV로 변환하기

콘솔에 `poetry run kaggle` 명령어를 실행합니다.

```console
$ poetry run kaggle
                                             id                                             title        date  rank        organization  teams
0   3d-object-detection-for-autonomous-vehicles  Lyft 3D Object Detection for Autonomous Vehicles  2022-02-04   119  Hanyang University    546
1        allstate-purchase-prediction-challenge                                               NaN  2022-04-18    19           SecuLayer   1566
2                 aptos2019-blindness-detection                                               NaN  2022-02-11  1850           SecuLayer   2928
3             bosch-production-line-performance                                               NaN  2022-05-12   351           SecuLayer   1370
4                     commonlitreadabilityprize                                               NaN  2022-06-02  1415           SecuLayer   3638
5                      dog-breed-identification                                               NaN  2022-02-08   191           SecuLayer   1280
6                               Don't Overfit!!                                               NaN  2022-02-10     2           SecuLayer   2316
7                        Herbarium 2020 - FGVC7                                               NaN  2022-02-10    59           SecuLayer    153
8          hpa-single-cell-image-classification                                               NaN  2022-02-21  3325           SecuLayer   3537
9                  jigsaw-toxic-severity-rating                                               NaN  2022-04-25   208           SecuLayer   2301
10          mercedes-benz-greener-manufacturing                                               NaN  2022-02-04    77  Hanyang University   3823
11    mlb-player-digital-engagement-forecasting                                               NaN  2022-06-09    92           SecuLayer    852
12           new-york-city-taxi-fare-prediction                                               NaN  2022-06-27   163           SecuLayer   1483
13                  petfinder-pawpularity-score                                               NaN  2022-02-16  3325           SecuLayer   3537
14                   plant-pathology-2021-fgvc8                                               NaN  2022-01-21   176           SecuLayer    626
15               plant-seedlings-classification                                               NaN  2022-02-08    30           SecuLayer    833
16                         quora-question-pairs                                               NaN  2022-05-03   913           SecuLayer   3295
17             santander-product-recommendation                                               NaN  2022-03-12   610           SecuLayer   1779
18                      shopee-product-matching                    Shopee - Price Match Guarantee  2021-03-18   398  Hanyang University   2426
19         Tabular Playground Series - Dec 2021                                               NaN  2022-02-08  1009           SecuLayer   1188
20         Tabular Playground Series - Jan 2022                                               NaN  2022-02-09  1420           SecuLayer   1591
21                   tmdb-box-office-prediction                                               NaN  2022-05-19   154           SecuLayer   1395
22     Transfer Learning on Stack Exchange Tags                                               NaN  2022-02-10   326           SecuLayer    380
23             traveling-santa-2018-prime-paths                Traveling Santa 2018 - Prime Paths  2022-01-24   841  Hanyang University   1867
24                   tweet-sentiment-extraction                                               NaN  2022-03-22   159           SecuLayer   2225
25     womens-machine-learning-competition-2019  Google Cloud & NCAA® ML Competition 2019-Women's  2022-03-14    70  Hanyang University    497
```

결과는 파일과 Console 모두에 출력합니다. 결과 파일은 `PROJECT_ROOT/outputs` 폴더 안에 `대회.csv` 이름으로 저장됩니다.
