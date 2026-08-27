from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RATINGS_PATH = Path("data/raw/ml-25m/ratings.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
POSITIVE_RATING_THRESHOLD = 4.0
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
MIN_INTERACTIONS_PER_USER = 10


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
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=MIN_INTERACTIONS_PER_USER,
        help=(
            "Minimum user history needed for per-user train/val/test splitting. "
            "Users below this threshold stay entirely in train."
        ),
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


def per_user_temporal_train_val_test_split(
    ratings: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    min_interactions_per_user: int = MIN_INTERACTIONS_PER_USER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")
    if min_interactions_per_user < 10:
        raise ValueError("min_interactions_per_user must be at least 10.")

    sorted_ratings = ratings.sort_values(
        by=["userId", "timestamp", "movieId"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = sorted_ratings.groupby("userId", sort=False)
    user_interaction_counts = grouped["movieId"].transform("size")
    user_positions = grouped.cumcount()

    eligible_users = user_interaction_counts.ge(min_interactions_per_user)
    train_counts = (user_interaction_counts * train_frac).astype("int64")
    val_counts = (user_interaction_counts * val_frac).astype("int64").clip(lower=1)
    val_ends = train_counts + val_counts

    # Users with too little history stay in train and are excluded from eval.
    train_mask = ~eligible_users | user_positions.lt(train_counts)
    val_mask = eligible_users & user_positions.ge(train_counts) & user_positions.lt(val_ends)
    test_mask = eligible_users & user_positions.ge(val_ends)

    train = sorted_ratings.loc[train_mask].copy()
    val = sorted_ratings.loc[val_mask].copy()
    test = sorted_ratings.loc[test_mask].copy()

    validate_per_user_temporal_order(train, val, test)

    return train, val, test


def validate_per_user_temporal_order(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    if train.empty or val.empty or test.empty:
        raise ValueError("Each split must contain at least one row.")

    train_users = set(train["userId"].unique())
    val_users = set(val["userId"].unique())
    test_users = set(test["userId"].unique())

    if not val_users.issubset(train_users):
        raise ValueError("Validation contains users with no training history.")
    if not test_users.issubset(train_users):
        raise ValueError("Test contains users with no training history.")

    train_max_timestamp = train.groupby("userId")["timestamp"].max()
    val_min_timestamp = val.groupby("userId")["timestamp"].min()
    val_max_timestamp = val.groupby("userId")["timestamp"].max()
    test_min_timestamp = test.groupby("userId")["timestamp"].min()

    if (train_max_timestamp.loc[list(val_users)] > val_min_timestamp).any():
        raise ValueError("Temporal leakage detected between train and validation.")

    if (val_max_timestamp.loc[list(test_users)] > test_min_timestamp).any():
        raise ValueError("Temporal leakage detected between validation and test.")


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
    train, val, test = per_user_temporal_train_val_test_split(
        labeled_ratings,
        min_interactions_per_user=args.min_interactions_per_user,
    )
    save_splits(train, val, test, args.output_dir)

    print(f"Saved {len(train):,} rows to {args.output_dir / 'train.csv'}")
    print(f"Saved {len(val):,} rows to {args.output_dir / 'val.csv'}")
    print(f"Saved {len(test):,} rows to {args.output_dir / 'test.csv'}")
    print(f"Validation users: {val['userId'].nunique():,}")
    print(f"Test users: {test['userId'].nunique():,}")


if __name__ == "__main__":
    main()
