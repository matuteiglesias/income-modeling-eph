# Labor bridge shadow — L2 model and L4 welfare handoff

Status: L1/L2/L3 real-data commissioning is active; L4 requires exact source-person identity.

This is a shadow path. It does not change the frozen income flagship or authorize
2022/23 predictive welfare.

## Scientific structure

The bridge never learns or samples a single hard `CONDACT` state.

It factorizes labor state into:

```text
P(active | X)
P(unemployed | active, X)
```

with `X` restricted by `configs/labor_bridge_feature_contract.yaml` to shared
pre-labor EPH/Census covariates.

Derived probabilities are:

```text
p_inactive   = 1 - p_active
p_unemployed = p_active * p_unemployed_given_active
p_employed   = p_active - p_unemployed
```

This makes the three-state simplex coherent by construction.

`ESTADO=0` individual non-response is not a supervised target. States 1/2/3/4
are used to train activity, and states 1/2 train conditional unemployment.

## L2 model comparison

The first model study deliberately compares only:

- logistic regression;
- HistGradientBoostingClassifier.

Both use the same shared covariates. Each should be run twice:

- `--weight-mode none`;
- `--weight-mode pondera`.

Cross-fitting is household-grouped by default. Every EPH probability used by
the income shadow path is OOF. The final two classifiers are then fitted on all
eligible EPH rows only for Census inference.

Example:

```bash
python scripts/14_labor_bridge_fit.py \
  --data /home/matias/data/harmonized-eph-2024-q3.parquet \
  --classifier-kind logistic \
  --weight-mode none \
  --output /home/matias/data/labor-bridge-2024-q3-logistic-unweighted

python scripts/14_labor_bridge_fit.py \
  --data /home/matias/data/harmonized-eph-2024-q3.parquet \
  --classifier-kind hist_gradient_boosting \
  --weight-mode pondera \
  --output /home/matias/data/labor-bridge-2024-q3-hgb-pondera
```

Repeat across the commissioning periods. Selection between specifications should
use held-out/OOF diagnostics plus transport behavior, not training fit.

## L3 raw Census scoring

This step is blocked on the real full-payload + harmonized Census frame.

When available:

```bash
python scripts/15_labor_bridge_score_census.py \
  --model /home/matias/data/<chosen-labor-release>/labor_bridge_model.joblib \
  --census /home/matias/data/<harmonized-census-frame>.parquet \
  --person-id-column sample_person_id \
  --calibration-domain-column eph_agglomerate_id \
  --output /home/matias/data/census-labor-raw-2024-q3
```

The raw release is then calibrated by `indice-pobreza-UBA`'s L3 calibrator.
Outside-EPH rows may be scored but are not benchmark-calibrated.

## L4 income shadow contract

`configs/feature_contract_labor_bridge_shadow.yaml` replaces the observed labor
feature block:

```text
CAT_INAC
CAT_OCUP
CONDACT
PP07G_59
```

with:

```text
labor_p_active
labor_p_employed
labor_p_unemployed
```

and explicitly forbids observed labor variables as predictors.

Attach an exact OOF probability release to the corresponding EPH modeling frame.

Source-backed modeling builds publish `modeling_identity.parquet` as a
non-predictor sidecar keyed by `row_id`. The sidecar carries the exact source
person-period key:

```text
CODUSU
NRO_HOGAR
COMPONENTE
ANO4
TRIMESTRE
```

Use it when identifiers are intentionally absent from the estimator-facing frame:

```bash
python scripts/16_build_labor_bridge_shadow_frame.py \
  --frame data/processed/modeling_dataset.parquet \
  --identity-sidecar data/processed/modeling_identity.parquet \
  --probabilities /home/matias/data/<labor-release>/eph_labor_probabilities_oof.parquet \
  --output data/processed/modeling_dataset_labor_bridge_shadow.parquet
```

The join fails unless both `row_id` coverage and person-period coverage are exact.
Row-order, nearest-feature, and fuzzy joins are forbidden.

### Legacy annual identity limitation

The tracked `EPHARG_annual_input_22..25.csv` artifacts predate this source
identity boundary. Their manifests include `CODUSU, ANO4, TRIMESTRE` but not
`NRO_HOGAR` or `COMPONENTE`. Therefore they cannot produce an exact identity
sidecar and cannot by themselves authorize the L4 OOF attachment.

The historical producer also joined household/person rows on the reduced key
`CODUSU, ANO4, TRIMESTRE, AGLOMERADO`. Before using the legacy artifact as a
regression oracle, quantify the raw EPH surface where one such reduced key spans
multiple true `NRO_HOGAR` values:

```bash
python scripts/17_audit_legacy_identity_gap.py \
  --individual /path/to/usu_individual_t124.txt \
  --individual /path/to/usu_individual_t224.txt \
  --individual /path/to/usu_individual_t324.txt \
  --individual /path/to/usu_individual_t424.txt \
  --output-json data/legacy-identity-gap-2024.json \
  --output-ambiguous-csv data/legacy-identity-gap-2024-ambiguous.csv
```

Do not reconstruct `NRO_HOGAR` or `COMPONENTE` from row order.

The shadow comparison must keep the current production P1-R untouched and report
at least:

- income OOF/test metrics;
- hurdle calibration;
- Telescope-B observed/point/predictive poverty and indigence;
- Telescope-C EPH↔Census transport;
- delta versus current P1-R.

Only after that comparison is reviewed should this contract be considered for
production or for 2022/23 predictive backcasting.


## Run the income shadow experiment

The experiment runner does not infer `baseline_feature_blocks` automatically.
Use the dedicated shadow experiment config so the final feature view contains
only `demographic + education + labor_bridge + housing_household`. This makes
the observed labor labels unavailable to the estimator even if they remain in
the diagnostic dataset.

Build the shadow processed dataset only after the processed frame has an exact
source identity sidecar:

```bash
python scripts/16_build_labor_bridge_shadow_frame.py \
  --frame data/processed/modeling_dataset.parquet \
  --identity-sidecar data/processed/modeling_identity.parquet \
  --probabilities /home/matias/data/<labor-release>/eph_labor_probabilities_oof.parquet \
  --output data/processed/modeling_dataset_labor_bridge_shadow.parquet
```

Then use the ordinary governed experiment runner:

```bash
python scripts/02_run_baseline_experiment.py \
  --config configs/experiment_labor_bridge_shadow.yaml \
  --feature-contract configs/feature_contract_labor_bridge_shadow.yaml \
  --allow-full-run
```

The existing split assignments remain valid because the shadow-frame join is
one-to-one and row-preserving. Do not use `--freeze-estimator`; this remains a
shadow comparison against the frozen flagship.
