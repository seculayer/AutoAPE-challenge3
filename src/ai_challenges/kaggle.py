import pandas as pd
import yaml

from .common import KAGGLE_PATH, RESULT_PATH, walk_filter_ext


def process_kaggle_competitions():
    pass


def main():
    metadata_files = walk_filter_ext("yaml", KAGGLE_PATH)

    competition_metadata = []

    for meta_file in metadata_files:
        with open(KAGGLE_PATH / meta_file, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

            competition_metadata.append(metadata)

    for competition in competition_metadata:
        if not "title" in competition.keys():
            print(competition.get("id"))

    df = pd.DataFrame.from_dict(competition_metadata)

    filtered_df = df[["id", "title", "date", "rank", "organization"]]

    filtered_df["teams"] = df["teams"].combine_first(df["team"]).astype(int)

    print(filtered_df)

    filtered_df.to_csv(RESULT_PATH / "kaggle.csv")


if __name__ == "__main__":
    main()
