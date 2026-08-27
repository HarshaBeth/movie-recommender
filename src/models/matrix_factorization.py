from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_ranking_metrics


UserId: TypeAlias = int
MovieId: TypeAlias = int

DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_VAL_PATH = Path("data/processed/val.csv")
DEFAULT_ARTIFACT_DIR = Path("artifacts/matrix_factorization")
POSITIVE_LABEL = "positive"
MIN_EVAL_K = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_NEGATIVES_PER_POSITIVE = 4


@dataclass
class IdMappings:
    user_id_to_idx: dict[UserId, int]
    movie_id_to_idx: dict[MovieId, int]
    idx_to_user_id: list[UserId]
    idx_to_movie_id: list[MovieId]


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    relevant_items_by_user: dict[UserId, set[MovieId]]
    stats: dict[str, int]


class BPRDataset(Dataset):
    def __init__(
        self,
        user_indices: Sequence[int],
        positive_movie_indices: Sequence[int],
        negative_movie_indices: np.ndarray,
    ) -> None:
        self.user_indices = torch.as_tensor(user_indices, dtype=torch.long)
        self.positive_movie_indices = torch.as_tensor(
            positive_movie_indices, dtype=torch.long
        )
        self.negative_movie_indices = torch.as_tensor(negative_movie_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_indices[idx],
            self.positive_movie_indices[idx],
            self.negative_movie_indices[idx],
        )


class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.movie_embeddings.weight, mean=0.0, std=0.05)

    def forward(self, user_indices: torch.Tensor, movie_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embeddings(user_indices)
        movie_vectors = self.movie_embeddings(movie_indices)
        return (user_vectors * movie_vectors).sum(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PyTorch matrix factorization recommender."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train.csv. Defaults to {DEFAULT_TRAIN_PATH}.",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help=f"Path to val.csv. Defaults to {DEFAULT_VAL_PATH}.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Directory for model artifacts. Defaults to {DEFAULT_ARTIFACT_DIR}.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension for users and movies.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Training mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of passes over train.csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Largest recommendation list size to generate for evaluation.",
    )
    parser.add_argument(
        "--eval-user-batch-size",
        type=int,
        default=256,
        help="Number of users scored at once during validation ranking.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--skip-popularity-comparison",
        action="store_true",
        help="Do not compute the popularity baseline comparison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for BPR negative sampling.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=DEFAULT_NEGATIVES_PER_POSITIVE,
        help="Number of sampled negative movies per positive interaction for BPR.",
    )
    return parser.parse_args()


def load_interactions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["userId", "movieId", "label"],
        dtype={"userId": "int32", "movieId": "int32", "label": "category"},
    )


def build_id_mappings(train: pd.DataFrame) -> IdMappings:
    """Create deterministic train-only raw ID to embedding-index mappings."""
    user_ids = sorted(int(user_id) for user_id in train["userId"].unique())
    movie_ids = sorted(int(movie_id) for movie_id in train["movieId"].unique())

    return IdMappings(
        user_id_to_idx={user_id: idx for idx, user_id in enumerate(user_ids)},
        movie_id_to_idx={movie_id: idx for idx, movie_id in enumerate(movie_ids)},
        idx_to_user_id=user_ids,
        idx_to_movie_id=movie_ids,
    )


def encode_interactions(
    train: pd.DataFrame, mappings: IdMappings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_indices = train["userId"].map(mappings.user_id_to_idx).astype("int64")
    movie_indices = train["movieId"].map(mappings.movie_id_to_idx).astype("int64")
    positive_mask = train["label"].eq(POSITIVE_LABEL).to_numpy(copy=True)
    return (
        user_indices.to_numpy(copy=True),
        movie_indices.to_numpy(copy=True),
        positive_mask,
    )


def sample_fallback_negatives(
    rng: np.random.Generator,
    num_movies: int,
    user_positive_movies: set[int],
    negatives_per_positive: int,
) -> np.ndarray:
    if len(user_positive_movies) >= num_movies:
        raise ValueError("Cannot sample a negative movie for a user with all movies positive.")

    negative_movie_indices = np.empty(negatives_per_positive, dtype=np.int64)
    for idx in range(negatives_per_positive):
        negative_movie_idx = int(rng.integers(num_movies))
        while negative_movie_idx in user_positive_movies:
            negative_movie_idx = int(rng.integers(num_movies))
        negative_movie_indices[idx] = negative_movie_idx

    return negative_movie_indices


def build_bpr_training_arrays(
    train: pd.DataFrame,
    mappings: IdMappings,
    seed: int,
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train-only (user, positive movie, negative movie) BPR triples."""
    rng = np.random.default_rng(seed)
    user_indices, movie_indices, positive_mask = encode_interactions(train, mappings)

    positive_user_indices = user_indices[positive_mask]
    positive_movie_indices = movie_indices[positive_mask]
    negative_user_indices = user_indices[~positive_mask]
    observed_negative_movie_indices = movie_indices[~positive_mask]

    negative_movies_by_user: dict[int, list[int]] = defaultdict(list)
    for user_idx, movie_idx in zip(negative_user_indices, observed_negative_movie_indices):
        negative_movies_by_user[int(user_idx)].append(int(movie_idx))

    positive_user_set = {int(user_idx) for user_idx in positive_user_indices}
    fallback_user_indices = positive_user_set - set(negative_movies_by_user)
    fallback_positive_movies_by_user: dict[int, set[int]] = defaultdict(set)
    if fallback_user_indices:
        for user_idx, movie_idx in zip(positive_user_indices, positive_movie_indices):
            if user_idx in fallback_user_indices:
                fallback_positive_movies_by_user[int(user_idx)].add(int(movie_idx))

    negative_movie_indices = np.empty(
        (len(positive_user_indices), negatives_per_positive),
        dtype=np.int64,
    )
    num_movies = len(mappings.idx_to_movie_id)

    for row_idx, user_idx in enumerate(positive_user_indices):
        user_idx = int(user_idx)
        user_negatives = negative_movies_by_user.get(user_idx)

        if user_negatives:
            sampled_indices = rng.integers(
                len(user_negatives),
                size=negatives_per_positive,
            )
            negative_movie_indices[row_idx] = [
                user_negatives[int(sampled_idx)] for sampled_idx in sampled_indices
            ]
        else:
            negative_movie_indices[row_idx] = sample_fallback_negatives(
                rng=rng,
                num_movies=num_movies,
                user_positive_movies=fallback_positive_movies_by_user[user_idx],
                negatives_per_positive=negatives_per_positive,
            )

    return positive_user_indices, positive_movie_indices, negative_movie_indices


def build_train_dataloader(
    train: pd.DataFrame,
    mappings: IdMappings,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    negatives_per_positive: int,
) -> DataLoader:
    user_indices, positive_movie_indices, negative_movie_indices = build_bpr_training_arrays(
        train=train,
        mappings=mappings,
        seed=seed,
        negatives_per_positive=negatives_per_positive,
    )
    dataset = BPRDataset(
        user_indices=user_indices,
        positive_movie_indices=positive_movie_indices,
        negative_movie_indices=negative_movie_indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: MatrixFactorizationModel,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for user_indices, positive_movie_indices, negative_movie_indices in train_loader:
            user_indices = user_indices.to(device)
            positive_movie_indices = positive_movie_indices.to(device)
            negative_movie_indices = negative_movie_indices.to(device)

            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = score_bpr_batch(
                model=model,
                user_indices=user_indices,
                positive_movie_indices=positive_movie_indices,
                negative_movie_indices=negative_movie_indices,
            )
            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            batch_size = user_indices.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        epoch_loss = total_loss / total_examples
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} - train loss: {epoch_loss:.6f}")

    return epoch_losses


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def score_bpr_batch(
    model: MatrixFactorizationModel,
    user_indices: torch.Tensor,
    positive_movie_indices: torch.Tensor,
    negative_movie_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    user_vectors = model.user_embeddings(user_indices)
    positive_movie_vectors = model.movie_embeddings(positive_movie_indices)
    negative_movie_vectors = model.movie_embeddings(negative_movie_indices)

    positive_scores = (user_vectors * positive_movie_vectors).sum(dim=1, keepdim=True)
    negative_scores = (user_vectors.unsqueeze(1) * negative_movie_vectors).sum(dim=2)
    return positive_scores, negative_scores


@torch.no_grad()
def score_user_movie_pairs(
    model: MatrixFactorizationModel,
    user_ids: Sequence[UserId],
    movie_ids: Sequence[MovieId],
    mappings: IdMappings,
    device: torch.device,
    batch_size: int = 65536,
) -> list[float]:
    """Score raw user/movie ID pairs with the trained embeddings."""
    if len(user_ids) != len(movie_ids):
        raise ValueError("user_ids and movie_ids must have the same length.")

    model.to(device)
    model.eval()
    scores: list[float] = []

    for start in range(0, len(user_ids), batch_size):
        end = start + batch_size
        user_batch = user_ids[start:end]
        movie_batch = movie_ids[start:end]

        user_indices = [
            mappings.user_id_to_idx[int(user_id)] for user_id in user_batch
        ]
        movie_indices = [
            mappings.movie_id_to_idx[int(movie_id)] for movie_id in movie_batch
        ]

        user_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)
        movie_tensor = torch.as_tensor(movie_indices, dtype=torch.long, device=device)
        logits = model(user_tensor, movie_tensor)
        scores.extend(logits.cpu().tolist())

    return scores


def build_user_interactions(interactions: pd.DataFrame) -> dict[UserId, set[MovieId]]:
    user_items: dict[UserId, set[MovieId]] = defaultdict(set)

    for user_id, movie_id in zip(
        interactions["userId"].to_numpy(),
        interactions["movieId"].to_numpy(),
    ):
        user_items[int(user_id)].add(int(movie_id))

    return dict(user_items)


def filter_warm_positive_eval(eval_data: pd.DataFrame, mappings: IdMappings) -> pd.DataFrame:
    positive_eval = eval_data.loc[eval_data["label"] == POSITIVE_LABEL]
    known_user_ids = set(mappings.user_id_to_idx)
    known_movie_ids = set(mappings.movie_id_to_idx)

    # Keep only warm-start positives the embedding model can actually score.
    return positive_eval.loc[
        positive_eval["userId"].isin(known_user_ids)
        & positive_eval["movieId"].isin(known_movie_ids)
    ]


def build_relevant_items(
    eval_data: pd.DataFrame, mappings: IdMappings
) -> dict[UserId, set[MovieId]]:
    return build_user_interactions(filter_warm_positive_eval(eval_data, mappings))


@torch.no_grad()
def recommend_top_k_for_user(
    model: MatrixFactorizationModel,
    user_id: UserId,
    mappings: IdMappings,
    train_interactions_by_user: dict[UserId, set[MovieId]],
    k: int,
    device: torch.device,
) -> list[MovieId]:
    """Generate top-K known movies, excluding movies seen in train."""
    if user_id not in mappings.user_id_to_idx:
        raise ValueError(f"Unknown userId {user_id}; no train embedding exists.")

    model.to(device)
    model.eval()
    user_idx = torch.tensor([mappings.user_id_to_idx[user_id]], device=device)
    user_vector = model.user_embeddings(user_idx)
    scores = user_vector @ model.movie_embeddings.weight.T
    scores = scores.squeeze(0)

    seen_movie_indices = [
        mappings.movie_id_to_idx[movie_id]
        for movie_id in train_interactions_by_user.get(user_id, set())
        if movie_id in mappings.movie_id_to_idx
    ]
    if seen_movie_indices:
        scores[seen_movie_indices] = -torch.inf

    top_k = min(k, scores.size(0))
    top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()
    return [mappings.idx_to_movie_id[movie_idx] for movie_idx in top_indices]


@torch.no_grad()
def recommend_for_users(
    model: MatrixFactorizationModel,
    user_ids: Iterable[UserId],
    mappings: IdMappings,
    train_interactions_by_user: dict[UserId, set[MovieId]],
    k: int,
    device: torch.device,
    user_batch_size: int,
) -> dict[UserId, list[MovieId]]:
    model.to(device)
    model.eval()
    recommendations_by_user: dict[UserId, list[MovieId]] = {}
    user_id_list = [int(user_id) for user_id in user_ids]
    movie_embeddings = model.movie_embeddings.weight

    for start in range(0, len(user_id_list), user_batch_size):
        batch_user_ids = user_id_list[start : start + user_batch_size]
        user_indices = [
            mappings.user_id_to_idx[user_id]
            for user_id in batch_user_ids
            if user_id in mappings.user_id_to_idx
        ]
        known_user_ids = [
            user_id for user_id in batch_user_ids if user_id in mappings.user_id_to_idx
        ]

        if not known_user_ids:
            continue

        user_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)
        scores = model.user_embeddings(user_tensor) @ movie_embeddings.T

        # Filter train interactions row-by-row before taking each user's top-K.
        for row_idx, user_id in enumerate(known_user_ids):
            seen_movie_indices = [
                mappings.movie_id_to_idx[movie_id]
                for movie_id in train_interactions_by_user.get(user_id, set())
                if movie_id in mappings.movie_id_to_idx
            ]
            if seen_movie_indices:
                scores[row_idx, seen_movie_indices] = -torch.inf

        top_k = min(k, scores.size(1))
        top_indices = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()

        for user_id, movie_indices in zip(known_user_ids, top_indices):
            recommendations_by_user[user_id] = [
                mappings.idx_to_movie_id[movie_idx] for movie_idx in movie_indices
            ]

    return recommendations_by_user


def evaluate_model(
    model: MatrixFactorizationModel,
    train: pd.DataFrame,
    eval_data: pd.DataFrame,
    mappings: IdMappings,
    k: int,
    device: torch.device,
    user_batch_size: int,
) -> EvaluationResult:
    train_interactions_by_user = build_user_interactions(train)
    relevant_items_by_user = build_relevant_items(eval_data, mappings)
    recommendations_by_user = recommend_for_users(
        model=model,
        user_ids=relevant_items_by_user.keys(),
        mappings=mappings,
        train_interactions_by_user=train_interactions_by_user,
        k=k,
        device=device,
        user_batch_size=user_batch_size,
    )

    metrics = evaluate_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        recall_ks=(10, 50),
        ndcg_ks=(10,),
    )

    positive_eval = eval_data.loc[eval_data["label"] == POSITIVE_LABEL]
    warm_positive_eval = filter_warm_positive_eval(eval_data, mappings)
    stats = {
        "positive_eval_rows": len(positive_eval),
        "warm_positive_eval_rows": len(warm_positive_eval),
        "positive_eval_users": positive_eval["userId"].nunique(),
        "evaluated_users": len(relevant_items_by_user),
    }

    return EvaluationResult(
        metrics=metrics,
        relevant_items_by_user=relevant_items_by_user,
        stats=stats,
    )


def evaluate_existing_popularity_baseline(
    train: pd.DataFrame,
    relevant_items_by_user: dict[UserId, set[MovieId]],
    k: int,
) -> dict[str, float] | None:
    try:
        from src.models.popularity_baseline import (
            build_user_interactions as build_popularity_user_interactions,
            rank_movies_by_positive_interactions,
            recommend_for_users as recommend_popular_for_users,
        )
    except ImportError:
        return None

    ranked_movies = rank_movies_by_positive_interactions(train)
    train_interactions_by_user = build_popularity_user_interactions(train)
    recommendations_by_user = recommend_popular_for_users(
        user_ids=relevant_items_by_user.keys(),
        ranked_movies=ranked_movies,
        train_interactions_by_user=train_interactions_by_user,
        k=k,
    )

    return evaluate_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        recall_ks=(10, 50),
        ndcg_ks=(10,),
    )


def save_artifacts(
    model: MatrixFactorizationModel,
    mappings: IdMappings,
    artifact_dir: Path,
    config: dict[str, int | float | str],
    train_losses: list[float],
    val_metrics: dict[str, float],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "num_users": len(mappings.idx_to_user_id),
        "num_movies": len(mappings.idx_to_movie_id),
        "embedding_dim": int(config["embedding_dim"]),
        "mappings": asdict(mappings),
        "config": config,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
    }
    torch.save(checkpoint, artifact_dir / "model.pt")

    metadata = {
        "num_users": checkpoint["num_users"],
        "num_movies": checkpoint["num_movies"],
        "embedding_dim": checkpoint["embedding_dim"],
        "config": config,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
    }
    with (artifact_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"Recall@10: {metrics['recall@10']:.6f}")
    print(f"Recall@50: {metrics['recall@50']:.6f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.6f}")
    print(f"MRR: {metrics['mrr']:.6f}")


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < MIN_EVAL_K:
        raise ValueError(f"--top-k must be at least {MIN_EVAL_K} to compute Recall@50.")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.eval_user_batch_size <= 0:
        raise ValueError("--eval-user-batch-size must be positive.")
    if args.negatives_per_positive <= 0:
        raise ValueError("--negatives-per-positive must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    print(f"Loading train data from {args.train_path}")
    train = load_interactions(args.train_path)
    print(f"Loading validation data from {args.val_path}")
    val = load_interactions(args.val_path)

    mappings = build_id_mappings(train)
    print(
        f"Training on {len(train):,} interactions, "
        f"{len(mappings.idx_to_user_id):,} users, "
        f"{len(mappings.idx_to_movie_id):,} movies"
    )

    train_loader = build_train_dataloader(
        train=train,
        mappings=mappings,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(
        f"Training BPR on {len(train_loader.dataset):,} positive pairs "
        f"with {args.negatives_per_positive} negatives each"
    )
    model = MatrixFactorizationModel(
        num_users=len(mappings.idx_to_user_id),
        num_movies=len(mappings.idx_to_movie_id),
        embedding_dim=args.embedding_dim,
    )

    print(f"Training on device: {device}")
    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    eval_result = evaluate_model(
        model=model,
        train=train,
        eval_data=val,
        mappings=mappings,
        k=args.top_k,
        device=device,
        user_batch_size=args.eval_user_batch_size,
    )

    print(
        "Validation coverage: "
        f"{eval_result.stats['evaluated_users']:,}/"
        f"{eval_result.stats['positive_eval_users']:,} positive-label users, "
        f"{eval_result.stats['warm_positive_eval_rows']:,}/"
        f"{eval_result.stats['positive_eval_rows']:,} positive-label rows"
    )
    print_metrics("Matrix factorization validation metrics:", eval_result.metrics)

    if not args.skip_popularity_comparison:
        popularity_metrics = evaluate_existing_popularity_baseline(
            train=train,
            relevant_items_by_user=eval_result.relevant_items_by_user,
            k=args.top_k,
        )
        if popularity_metrics is not None:
            print_metrics(
                "Popularity baseline validation metrics on the same warm-start users:",
                popularity_metrics,
            )

    config = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "loss": "bpr",
        "optimizer": "Adam",
        "negative_sampling": "observed_train_negative_per_user_with_non_positive_fallback",
        "negatives_per_positive": args.negatives_per_positive,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "top_k": args.top_k,
        "device": str(device),
        "seed": args.seed,
    }
    save_artifacts(
        model=model,
        mappings=mappings,
        artifact_dir=args.artifact_dir,
        config=config,
        train_losses=train_losses,
        val_metrics=eval_result.metrics,
    )
    print(f"Saved model checkpoint and metadata to {args.artifact_dir}")


if __name__ == "__main__":
    main()
