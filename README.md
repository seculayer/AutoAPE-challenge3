# AI Competitions 3

Competitions 3차년도(2022)

## Contribute

1. [kaggle](#kaggle-구조)/dacon 안에 구조를 맞추어서, 참가한 대회 내용을 작성합니다.
2. Pull Request를 작성합니다.

## Kaggle 구조

Kaggle 폴더 구조입니다.

```console
kaggle/
└── {kaggle-competition-id}/
   ├── metadata.yaml
   ├── README.md
   └── *
```

1. Kaggle 대회 ID로 폴더를 생성합니다.

    Kaggle 대회 ID는 URL에서 찾을 수 있습니다.

    예시 - kaggle 대회 URL이 `https://www.kaggle.com/c/acea-water-prediction`이면, `acea-water-prediction` 부분이 대회 ID입니다.

2. `metadata.yaml`과 `README.md`를 작성합니다.

    metadata에는 아래와 같은 내용들이 있어야합니다.

    ```yaml
    version: 2
    id: kaggle-competition-id
    score: 캐글 대회 점수
    teams: 참가한 총 팀 수
    rank: 등수
    date: 대회에 제출한 날짜
    organization:
    team:
      name: 팀 이름
      members:
        - id: Kaggle ID
          name: 이름
        - id: Kaggle ID
          name: 이름
    ```

    팀 참여 예시.

    ```yaml
    version: 2
    id: titanic
    score: 0.817
    teams: 1000
    rank: 100
    date: 2021-05-20
    organization: Hanyang University
    team:
      name: "Team Example"
      members:
        - id: User1
          name: Jone
        - id: User2
          name: 홍길동
    ```

    개인 참여 예시

    ```yaml
    version: 2
    id: titanic
    score: 0.817
    teams: 1000
    rank: 100
    date: 2021-05-20
    organization: SecuLayer
    team:
      name: User1
      members:
        - id: User1
          name: Jone
    ```

    `README.md`에는 다음과 같은 내용이 있어야 합니다.

    - 결과 요약
    - 리더보드 이미지
      - 대회 점수 이미지
      - 등수 이미지
    - 알고리즘, 문제 해결 방법
    - 팀 참여 시 역할 및 기여도
    - 참고 자료
    - 기타

3. 작성했던 코드들을 생성했던 폴더에 넣습니다.
    - colab, jupyter 등을 사용했다면 `ipynb`을 그대로 올려도 됩니다.
4. Commit 하기
    - 팀 참여 시 commit 본문에 마지막에 `Co-authored-by: name <name@example.com>`를 추가하면, 한 커밋에 여러 작성자를 추가할 수 있습니다.
    - 팀 참여 `git commit` 커맨드 예제

    ```sh
     git commit -m "Refactor usability tests.
    >
    >
    Co-authored-by: name <name@example.com>
    Co-authored-by: another-name <another-name@example.com>"
    ```

    - 참고자료 : [Creating a commit with multiple authors](https://docs.github.com/en/github/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors)

## 참고

- [YAML](https://yaml.org/)
  - [YAML - 위키백과](https://ko.wikipedia.org/wiki/YAML)
- [Pro Git: 6.2 GitHub - GitHub 프로젝트에 기여하기](https://git-scm.com/book/ko/v2/GitHub-GitHub-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%EC%97%90-%EA%B8%B0%EC%97%AC%ED%95%98%EA%B8%B0)
