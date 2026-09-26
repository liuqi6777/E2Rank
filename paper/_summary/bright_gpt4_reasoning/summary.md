# BRIGHT GPT-4 query results

Complete: 30/30 runs. Query set: `gpt4-reasoning`.

Scores: nDCG@10 × 100. Avg. is the unweighted mean over 12 subsets. SD is sample SD across seeds.
Group statistics require every expected run and all 12 subsets. E0 is a single run; its SD is omitted.

| Backbone | Method | Complete | Avg. | Sample SD |
| --- | --- | ---: | ---: | ---: |
| 4b | E0 | 1/1 | 22.01 | — |
| 4b | InfoNCE | 3/3 | 30.29 | 0.35 |
| 4b | LambdaLoss | 3/3 | 25.69 | 0.25 |
| 4b | RELER | 3/3 | 31.07 | 0.39 |
| 06b | E0 | 1/1 | 20.38 | — |
| 06b | InfoNCE | 3/3 | 26.51 | 0.10 |
| 06b | LambdaLoss | 3/3 | 23.97 | 0.36 |
| 06b | RELER | 3/3 | 27.15 | 0.04 |
| bge-m3 | E0 | 1/1 | 21.45 | — |
| bge-m3 | InfoNCE | 3/3 | 20.62 | 0.09 |
| bge-m3 | LambdaLoss | 3/3 | 19.13 | 0.10 |
| bge-m3 | RELER | 3/3 | 23.84 | 0.07 |

## Pending / invalid runs

All 30 expected evaluations are complete.
