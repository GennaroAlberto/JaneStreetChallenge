# Benchmark results

Generated: 2026-05-28 20:27:00

| Model | params | fit (s) | R² static | R² online | Δ online |
|---|---:|---:|---:|---:|---:|
| lstm_modelr | 938,889 | 14757 | +0.00637 | +0.00909 | +0.00272 |
| gru_modelr | 704,265 | 9640 | +0.00763 | +0.00893 | +0.00130 |
| xgb | -1 | 964 | +0.00684 | — | — |

R² = competition weighted-R² (1 − ΣwΔ² / Σwy²). Higher is better. The 8th place LB score was 0.0112.