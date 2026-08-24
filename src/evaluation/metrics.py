from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from math import log2
from statistics import mean
from typing import Hashable


UserId = Hashable
ItemId = Hashable


def recall_at_k(
    recommendations: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Compute Recall@K for one user."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    recommended = set(recommendations[:k])
    return len(recommended.intersection(relevant)) / len(relevant)


def ndcg_at_k(
    recommendations: Sequence[ItemId],
    relevant_items: Collection[ItemId],
    k: int,
) -> float:
    """Compute binary-relevance NDCG@K for one user."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommendations[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(
    recommendations: Sequence[ItemId],
    relevant_items: Collection[ItemId],
) -> float:
    """Compute reciprocal rank for one user."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    for rank, item_id in enumerate(recommendations, start=1):
        if item_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_recall_at_k(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    relevant_items_by_user: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    return _mean_over_users(
        recommendations_by_user,
        relevant_items_by_user,
        lambda recommendations, relevant: recall_at_k(recommendations, relevant, k),
    )


def mean_ndcg_at_k(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    relevant_items_by_user: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    return _mean_over_users(
        recommendations_by_user,
        relevant_items_by_user,
        lambda recommendations, relevant: ndcg_at_k(recommendations, relevant, k),
    )


def mean_reciprocal_rank(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    relevant_items_by_user: Mapping[UserId, Collection[ItemId]],
) -> float:
    return _mean_over_users(
        recommendations_by_user,
        relevant_items_by_user,
        reciprocal_rank,
    )


def evaluate_ranking_metrics(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    relevant_items_by_user: Mapping[UserId, Collection[ItemId]],
    recall_ks: Sequence[int] = (10, 50),
    ndcg_ks: Sequence[int] = (10,),
) -> dict[str, float]:
    """Evaluate recommendations against users with at least one relevant item."""
    metrics: dict[str, float] = {}

    for k in recall_ks:
        metrics[f"recall@{k}"] = mean_recall_at_k(
            recommendations_by_user,
            relevant_items_by_user,
            k,
        )

    for k in ndcg_ks:
        metrics[f"ndcg@{k}"] = mean_ndcg_at_k(
            recommendations_by_user,
            relevant_items_by_user,
            k,
        )

    metrics["mrr"] = mean_reciprocal_rank(
        recommendations_by_user,
        relevant_items_by_user,
    )
    return metrics


def _mean_over_users(
    recommendations_by_user: Mapping[UserId, Sequence[ItemId]],
    relevant_items_by_user: Mapping[UserId, Collection[ItemId]],
    metric_fn,
) -> float:
    scores = [
        metric_fn(recommendations_by_user.get(user_id, ()), relevant_items)
        for user_id, relevant_items in relevant_items_by_user.items()
        if relevant_items
    ]
    return mean(scores) if scores else 0.0
