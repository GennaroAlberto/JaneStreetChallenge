# Benchmark results

Generated: 2026-05-15 10:04:42

| Model | params | fit (s) | R² static | R² online | Δ online |
|---|---:|---:|---:|---:|---:|
| lstm_modelr | 938,889 | 11640 | +0.00425 | +0.00798 | +0.00374 |
| gru_modelr | 704,265 | 10577 | +0.00497 | +0.00749 | +0.00253 |
| gru | 120,193 | 5408 | +0.00246 | +0.00749 | +0.00502 |
| xgb | -1 | 454 | +0.00587 | — | — |
| transformer | 260,257 | 5685 | +0.00474 | +0.00500 | +0.00026 |
| sig_transformer | 265,633 | 6598 | +0.00485 | +0.00436 | -0.00049 |
| mlp | 458,753 | 4024 | -0.00258 | — | — |

R² = competition weighted-R² (1 − ΣwΔ² / Σwy²). Higher is better. The 8th place LB score was 0.0112.