# Movie Recommender

A MovieLens 25M recommender-system project focused on building and evaluating
ranking models from implicit positive/negative feedback.

## Progress So Far

- Downloaded and explored the MovieLens 25M dataset.
- Created temporal train/validation/test splits from `ratings.csv`:
  - `train.csv`: earliest 80% of ratings
  - `val.csv`: next 10% of ratings
  - `test.csv`: latest 10% of ratings
- Added a binary label:
  - `positive` for ratings `>= 4.0`
  - `negative` for ratings `< 4.0`
- Implemented ranking metrics in `src/evaluation/metrics.py`:
  - `Recall@K`
  - `NDCG@K`
  - `MRR`
- Implemented a popularity-baseline recommender in
  `src/models/popularity_baseline.py`.

The split is chronological to avoid temporal data leakage. The popularity model
uses only `data/processed/train.csv` to rank movies by positive interaction
count, then evaluates first on `data/processed/val.csv`.

## Popularity Baseline Results

Validation results:

| Metric | Score |
| --- | ---: |
| Recall@10 | 0.058249 |
| Recall@50 | 0.173999 |
| NDCG@10 | 0.294354 |
| MRR | 0.508800 |

Run the baseline with:

```bash
python src/models/popularity_baseline.py
```

Local raw and processed data files are ignored by Git because the MovieLens 25M
CSV files are large and reproducible.
