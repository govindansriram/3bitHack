What to measure for 3-bit work
- Distribution shape: absmax, std, kurtosis, and the fraction of weights beyond 6σ, per tensor type and per layer.
- Spread within groups: how much the range varies across output channels and across 128-weight groups.
- Quantization error: relative error after simple 3-bit rounding (group size 128, symmetric vs asymmetric), which shows which layers and projections suffer most.
- Differences between experts: do some experts have more extreme outliers than others?