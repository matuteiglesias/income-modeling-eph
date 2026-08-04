# Flagship candidate-release readiness report

## Decision

**Ready for candidate consumption within the declared EPH inference boundary.** The release is
not reviewed or approved, is not Census-compatible, and is not suitable for poverty computation.

## Frozen identities

- Annual candidate releases: `artifact:research.eph-annual-preprocessed@1+2022` through `+2025`.
- Annual manifest SHA-256 values: `5fb51217f510779a2af40999e201a606053c28469172012d56737044f3dfecc6`,
  `cbb2d56c6a74aa17253c28eada9a8cf49c038dd42f417a2268fd507ab2e647b3`,
  `3fb83e08ec0ead95bd0edca4621c902fcad63cd8167901f26e605bb25b89ec60`, and
  `dc22e6a0f2f98bbae29d42d33369a422e1bb1863947c82c8a55bf351330123b9`.
- Input lock: `model-input-lock-00eb4d82075e6d01`; lock-file SHA-256
  `54ec31d95726764c7106f5eb77b7ade6549e4b8565c72aad3bdd3d5f7ac005b4`.
- Candidate release: `eph-income-model-00eb4d82075e6d01`.

## Run resolution and training

The initial inventory found no canonical run directories. After the annual preflight, canonical
dataset build, and lock creation passed, the exact governed command
`make run-hgb-quick-benchmark-freeze` trained
`reports/runs/hgb_quick_benchmark_v1_20260804T095654Z`. No other experiment was trained. The
identity resolver subsequently selected that run using config, feature-contract, processed-data,
split, experiment, required-artifact, and convenience-copy checks rather than recency.

The fitted `HistGradientBoostingRegressor` uses learning rate `0.05`, 200 iterations, 63 leaf
nodes, minimum leaf size 100, zero L2 regularization, disabled early stopping, squared-error loss,
and random state 42. Its observed log-scale metrics are CV R² 0.56204, test R² 0.58034 / MAE
0.18005, and validation R² 0.57656 / MAE 0.18058. These are predictive measurements on the
controlled pooled EPH-derived split, not causal or official-income claims.

## Release artifacts

The release contains a hashed fitted pipeline (`8feeeb81…8271`), input lock, exact dependency
snapshot, inference contract, bounded prediction fixture, model comparison, diagnostics, and test
and validation predictions. Its release manifest content identity is
`5cab9ea82417134a8b5b23801c1643a9946d07a80017c6ae533114da7383e241`.
The fitted joblib pipeline is trusted-local only: validators check the envelope and binary hash
before loading, and consumers must never load it from an untrusted source.

## Inference semantics

The primary output is `predicted_logP47T` on the `log10_ars` scale, exactly `log10(P47T)`. There
is no `+1` correction. The release does not emit a mechanically inverse-transformed income;
`10**prediction` would require explicit labeling and does not correct retransformation bias.
Input columns must exactly match the ordered EPH feature contract (apart from `row_id`), and row
identity/order is preserved. Census mode is rejected.

## Evidence and limitations

The selected flagship run passes all 13 registry-required artifact checks, including test and
validation diagnostics. The broader `thesis_core` evidence gate reports the flagship 13/13 but
also reports the absent supporting `baseline_income_prediction_v1` and
`hgb_quick_clean_geo_v1` runs; the packet explicitly prohibited rerunning them merely for this
freeze.

Historical raw-source reproduction, upstream source hashes, the historical monetary/price-series
reference, and a true unique person key remain unresolved. No Census inference/adaptation has
been approved. The release therefore remains `candidate`.

No tuning, target/split/feature/model change, poverty computation, Census inference, sibling-repo
execution, external binary publication, or promotion beyond candidate occurred.
