from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RATINGS_PATH = Path("data/raw/ml-25m/ratings.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
POSITIVE_RATING_THRESHOLD = 4.0
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chronological train/val/test splits for MovieLens ratings."
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
        help=f"Path to ratings.csv. Defaults to {DEFAULT_RATINGS_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where split CSVs will be written. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=POSITIVE_RATING_THRESHOLD,
        help="Ratings greater than or equal to this value are labeled positive.",
    )
    return parser.parse_args()


def validate_ratings_columns(ratings: pd.DataFrame) -> None:
    required_columns = {"userId", "movieId", "rating", "timestamp"}
    missing_columns = required_columns.difference(ratings.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ratings.csv is missing required columns: {missing}")


def add_rating_label(
    ratings: pd.DataFrame, positive_threshold: float
) -> pd.DataFrame:
    labeled = ratings.copy()
    labeled["label"] = labeled["rating"].ge(positive_threshold).map(
        {True: "positive", False: "negative"}
    )
    return labeled


def temporal_train_val_test_split(
    ratings: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")

    sorted_ratings = ratings.sort_values(
        by=["timestamp", "userId", "movieId"],
        kind="mergesort",
    ).reset_index(drop=True)

    n_ratings = len(sorted_ratings)
    train_end = int(n_ratings * train_frac)
    val_end = train_end + int(n_ratings * val_frac)

    train = sorted_ratings.iloc[:train_end].copy()
    val = sorted_ratings.iloc[train_end:val_end].copy()
    test = sorted_ratings.iloc[val_end:].copy()

    validate_temporal_order(train, val, test)

    return train, val, test


def validate_temporal_order(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    if train.empty or val.empty or test.empty:
        raise ValueError("Each split must contain at least one row.")

    if train["timestamp"].max() > val["timestamp"].min():
        raise ValueError("Temporal leakage detected: train contains validation future data.")

    if val["timestamp"].max() > test["timestamp"].min():
        raise ValueError("Temporal leakage detected: validation contains test future data.")


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    args = parse_args()

    ratings = pd.read_csv(args.ratings_path)
    validate_ratings_columns(ratings)

    labeled_ratings = add_rating_label(
        ratings=ratings,
        positive_threshold=args.positive_threshold,
    )
    train, val, test = temporal_train_val_test_split(labeled_ratings)
    save_splits(train, val, test, args.output_dir)

    print(f"Saved {len(train):,} rows to {args.output_dir / 'train.csv'}")
    print(f"Saved {len(val):,} rows to {args.output_dir / 'val.csv'}")
    print(f"Saved {len(test):,} rows to {args.output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
