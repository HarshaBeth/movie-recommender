# Movie Recommender

A MovieLens 25M recommender-system project focused on building and evaluating
ranking models from implicit positive/negative feedback.

## Progress So Far

- Downloaded and explored the MovieLens 25M dataset.
- Created per-user chronological train/validation/test splits from `ratings.csv`:
  - `train.csv`: earliest 80% of each eligible user's ratings
  - `val.csv`: next 10% of each eligible user's ratings
  - `test.csv`: latest 10% of each eligible user's ratings
- Added a binary label:
  - `positive` for ratings `>= 4.0`
  - `negative` for ratings `< 4.0`
- Fixed the original global timestamp split after finding it produced many
  validation/test users with no training history.
- Implemented ranking metrics in `src/evaluation/metrics.py`:
  - `Recall@K`
  - `NDCG@K`
  - `MRR`
- Implemented a popularity-baseline recommender in
  `src/models/popularity_baseline.py`.
- Implemented a PyTorch matrix factorization recommender in
  `src/models/matrix_factorization.py`.

Users with fewer than 10 interactions stay entirely in training so evaluation is
mostly warm-start. The current split sizes are:

| Split | Rows |
| --- | ---: |
| Train | 19,936,012 |
| Validation | 2,431,061 |
| Test | 2,633,022 |

Current validation warm-start coverage for matrix factorization:

```text
150,405 / 150,407 positive-label users
1,119,379 / 1,120,331 positive-label rows
```

## Experiment Log

Initial popularity baseline on the original global timestamp split:

| Metric | Score |
| --- | ---: |
| Recall@10 | 0.058249 |
| Recall@50 | 0.173999 |
| NDCG@10 | 0.294354 |
| MRR | 0.508800 |

The first matrix factorization experiments exposed a split issue: only
`5,077 / 17,847` validation users with positive interactions were warm-start.
The splitter was changed from global chronological slicing to per-user
chronological slicing.

Results after fixing the split, evaluated on `val.csv` and compared against the
popularity baseline on the same warm-start users:

| Model | Loss | Negatives per Positive | Recall@10 | Recall@50 | NDCG@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Popularity baseline | Count positives | N/A | 0.059504 | 0.174505 | 0.052038 | 0.098756 |
| Matrix factorization | BCEWithLogitsLoss | N/A | 0.048067 | 0.114238 | 0.043169 | 0.087594 |
| Matrix factorization | BPR | 1 | 0.049091 | 0.135035 | 0.045007 | 0.092768 |
| Matrix factorization | BPR | 4 | 0.050255 | 0.137480 | 0.045765 | 0.093860 |

The 1:4 BPR setup improved over BCE and 1-negative BPR, especially on
`Recall@50`, but it is still below the popularity baseline.

## Run Commands

Regenerate the data splits:

```bash
python src/data/split.py
```

Run the popularity baseline:

```bash
python src/models/popularity_baseline.py
```

Train and evaluate matrix factorization:

```bash
python src/models/matrix_factorization.py
```

Local raw and processed data files are ignored by Git because the MovieLens 25M
CSV files are large and reproducible.
