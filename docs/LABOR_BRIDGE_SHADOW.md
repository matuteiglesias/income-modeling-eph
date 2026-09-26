# Labor bridge shadow — L2 model and L4 welfare handoff

Status: software-ready; real EPH/Census runs remain local.

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

Attach an exact OOF probability release to the corresponding EPH modeling frame:

```bash
python scripts/16_build_labor_bridge_shadow_frame.py \
  --frame /home/matias/data/<income-model-frame>.parquet \
  --probabilities /home/matias/data/<labor-release>/eph_labor_probabilities_oof.parquet \
  --output /home/matias/data/<income-model-frame>-labor-shadow.parquet
```

The join fails unless person-period coverage is exact.

The shadow comparison must keep the current production P1-R untouched and report
at least:

- income OOF/test metrics;
- hurdle calibration;
- Telescope-B observed/point/predictive poverty and indigence;
- Telescope-C EPH↔Census transport;
- delta versus current P1-R.

Only after that comparison is reviewed should this contract be considered for
production or for 2022/23 predictive backcasting.
