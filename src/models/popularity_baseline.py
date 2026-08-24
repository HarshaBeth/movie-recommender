from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeAlias

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_ranking_metrics


UserId: TypeAlias = int
MovieId: TypeAlias = int

DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_EVAL_PATH = Path("data/processed/val.csv")
DEFAULT_TOP_K = 50
POSITIVE_LABEL = "positive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-only popularity baseline on MovieLens splits."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train.csv. Defaults to {DEFAULT_TRAIN_PATH}.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help=f"Path to validation or test split. Defaults to {DEFAULT_EVAL_PATH}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Largest recommendation list size to generate per user.",
    )
    return parser.parse_args()


def load_interactions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["userId", "movieId", "label"],
        dtype={"userId": "int32", "movieId": "int32", "label": "category"},
    )


def rank_movies_by_positive_interactions(train: pd.DataFrame) -> list[MovieId]:
    """Rank movies by positive interaction count using train data only."""
    positive_train = train.loc[train["label"] == POSITIVE_LABEL]

    popularity = (
        positive_train.groupby("movieId", observed=True)
        .size()
        .reset_index(name="positive_interactions")
        .sort_values(
            by=["positive_interactions", "movieId"],
            ascending=[False, True],
            kind="mergesort",
        )
    )

    return popularity["movieId"].astype(int).tolist()


def get_top_k_popular_movies(train: pd.DataFrame, k: int) -> list[MovieId]:
    """Return the global top-K most popular movies from train positives."""
    return rank_movies_by_positive_interactions(train)[:k]


def build_user_interactions(interactions: pd.DataFrame) -> dict[UserId, set[MovieId]]:
    """Map each user to all movies they interacted with in a split."""
    user_items: dict[UserId, set[MovieId]] = defaultdict(set)

    for user_id, movie_id in zip(
        interactions["userId"].to_numpy(),
        interactions["movieId"].to_numpy(),
    ):
        user_items[int(user_id)].add(int(movie_id))

    return dict(user_items)


def build_relevant_items(eval_data: pd.DataFrame) -> dict[UserId, set[MovieId]]:
    """Use only positive eval interactions as relevant items."""
    return build_user_interactions(eval_data.loc[eval_data["label"] == POSITIVE_LABEL])


def recommend_for_user(
    ranked_movies: Sequence[MovieId],
    seen_movies: set[MovieId],
    k: int,
) -> list[MovieId]:
    recommendations: list[MovieId] = []

    # Remove train interactions so we do not recommend movies the user already saw.
    for movie_id in ranked_movies:
        if movie_id in seen_movies:
            continue

        recommendations.append(movie_id)
        if len(recommendations) == k:
            break

    return recommendations


def recommend_for_users(
    user_ids: Iterable[UserId],
    ranked_movies: Sequence[MovieId],
    train_interactions_by_user: dict[UserId, set[MovieId]],
    k: int,
) -> dict[UserId, list[MovieId]]:
    return {
        user_id: recommend_for_user(
            ranked_movies=ranked_movies,
            seen_movies=train_interactions_by_user.get(user_id, set()),
            k=k,
        )
        for user_id in user_ids
    }


def evaluate_popularity_baseline(
    train_path: Path = DEFAULT_TRAIN_PATH,
    eval_path: Path = DEFAULT_EVAL_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    train = load_interactions(train_path)
    eval_data = load_interactions(eval_path)

    ranked_movies = rank_movies_by_positive_interactions(train)
    train_interactions_by_user = build_user_interactions(train)
    relevant_items_by_user = build_relevant_items(eval_data)

    recommendations_by_user = recommend_for_users(
        user_ids=relevant_items_by_user.keys(),
        ranked_movies=ranked_movies,
        train_interactions_by_user=train_interactions_by_user,
        k=top_k,
    )

    return evaluate_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        recall_ks=(10, 50),
        ndcg_ks=(10,),
    )


def main() -> None:
    args = parse_args()
    metrics = evaluate_popularity_baseline(
        train_path=args.train_path,
        eval_path=args.eval_path,
        top_k=args.top_k,
    )

    print(f"Recall@10: {metrics['recall@10']:.6f}")
    print(f"Recall@50: {metrics['recall@50']:.6f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.6f}")
    print(f"MRR: {metrics['mrr']:.6f}")


if __name__ == "__main__":
    main()
