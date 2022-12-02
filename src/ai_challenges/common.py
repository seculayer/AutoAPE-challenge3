import glob
from os import PathLike
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

KAGGLE_PATH = PROJECT_ROOT / "kaggle"
DACON_PATH = PROJECT_ROOT / "dacon"

RESULT_PATH = PROJECT_ROOT / "outputs"


def walk_filter_ext(ext: str, path: PathLike):
    return glob.glob(f"*/*.{ext}", root_dir=path)


if not RESULT_PATH.exists():
    RESULT_PATH.mkdir()
