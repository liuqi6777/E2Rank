# Embedding analysis

Scores are in [0, 1]; nDCG uses linear relevance gains.
Reference: E0. Unlabelled candidates are not verified semantic negatives.

| Model | Domains | Macro nDCG@10 | Macro MRR@10 |
|---|---:|---:|---:|
| E0 | 4 | 0.213018 | 0.250948 |
| InfoNCE | 4 | 0.299313 | 0.352425 |
| LambdaLoss | 4 | 0.239689 | 0.280330 |
| RELER | 4 | 0.329914 | 0.401073 |
| Listwise | 4 | 0.312063 | 0.378122 |

Per-domain retrieval: [retrieval.csv](retrieval.csv).
Paired changes: [paired_queries.csv](paired_queries.csv); reference-rank bins: [rank_bins.csv](rank_bins.csv).
Geometry: [geometry.csv](geometry.csv); fixed document pairs are in each subset's geometry directory.
Perturbations: [perturb.csv](perturb.csv). Candidate-scope results are local reranking, not full-corpus retrieval.

No significance claim is made from a single training seed. Monte Carlo SE measures sampling error only.
Absent optional stages produce empty CSVs. Exact ties use descending document ID.
