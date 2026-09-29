# Longitudinal EPH 2017-Q1..2026-Q1 — C2 implementation contract

This document records the implemented cloud-side contract for the new longitudinal EPH lane. It does not change the frozen 2022-Q1..2025-Q1 flagship inputs or their historical cohort semantics.

## Product boundary

The builder emits an immutable artifact with contract:

```text
research.eph-longitudinal-analysis-frame/v1
```

It consumes:

- exactly 37 already-pinned `publicdata.eph-microdata@1` parents, `2017-Q1..2026-Q1`;
- one immutable `research.argentina-monetary-conversion/v1` release from `IPC-Argentina`;
- `configs/longitudinal_eph_harmonization_v1.json`.

Raw EPH acquisition remains owned by `microdatos-EPH-INDEC`. This repository only verifies and locks consumer-side parents.

## Exact parent lock

`scripts/17_build_longitudinal_parent_ledger.py` scans a root of already-pinned releases and writes `eph-longitudinal-parent-lock/v1`.

Every entry records period, release ID, parent-lock checksum, producer-manifest checksum, source-manifest/archive checksums, exact household/individual table checksums, row counts, ordered columns and schema fingerprints. Missing quarters, duplicate quarters, checksum drift or a nearby-quarter substitution fail closed.

The lock contains no absolute machine path, so its identity is portable between the cloud runner and L2 local roots.

## Schema drift and harmonization

The harmonization policy deliberately starts conservative:

- required identity/period/geography/income fields must exist;
- at least one source-faithful labor state (`ESTADO` or `CONDACT`) must exist;
- no undocumented semantic rename is allowed;
- optional fields are unioned across quarters and are blank only when absent from that quarter's source schema;
- target-derived `AGLO_rk`, `Reg_rk` and `logP47T` are rejected.

The release includes `schema_inventory.csv`, with period-by-role fingerprints plus added/removed/reordered columns and the applied harmonization action. L2 is responsible for examining the real 37-quarter inventory and isolating any historical quarter that violates the stable contract rather than inventing an alias.

## Observation and panel identity

The ordinary observation identities are period-qualified:

```text
household_observation_id = period | CODUSU | NRO_HOGAR
row_id                   = period | CODUSU | NRO_HOGAR | COMPONENTE
```

Repeated-wave candidates are separate:

```text
panel_dwelling_id            = CODUSU
panel_household_id           = CODUSU | NRO_HOGAR
person_linkage_candidate_id  = CODUSU | NRO_HOGAR | COMPONENTE
```

`panel_links.csv` audits consecutive candidate observations using elapsed quarters and available `CH04`/`CH06` demographic consistency. Conflicts are emitted explicitly. The contract never promotes this candidate key to permanent person identity; it is suitable only for short-horizon EPH persistence experiments. `panel_households.csv` summarizes repeated household candidates.

## Source-faithful person/household frame

Every source individual row is retained. The person output preserves the union of native individual columns and adds household context with an `HH__` prefix after a validated many-person-to-one-household join. The builder checks row period, unique within-period identities and orphan persons.

Canonical metadata includes `period`, `region_id`, exact source release/row identity, explicit survey-weight aliases, exceptional-period policy fields, and the repeated-wave candidate IDs. Survey weights are preserved but never applied by this builder.

## Monetary semantics

The new lane preserves both:

```text
P47T_nominal
P47T_real
```

The caller must supply `--monetary-reference-period YYYY-MM-01`; there is intentionally no historical January-2016 default. Quarterly source amounts use the deterministic Q1→February, Q2→May, Q3→August, Q4→November mapping because EPH only identifies the quarter here.

The current IPC conversion artifact supplies factors from each month to its producer base. C2 composes two factors from the same immutable release:

```text
factor(source quarter month -> chosen reference month)
  = factor(source quarter month -> producer base)
    / factor(chosen reference month -> producer base)
```

No rounding is applied. `monetary_lineage.csv` records the exact release, source month, chosen target month, producer base, factor and coverage class. Zero P47T stays zero. Negative source codes/values remain in `P47T_nominal` but are not silently converted to a real-income amount.

Approved monetary releases are required by default. `--allow-candidate-conversion` is an explicit bounded-evidence escape hatch and does not promote the candidate release.

## Exceptional periods

All observations remain in the artifact. The following metadata are fixed:

```text
2020-Q2  pandemic_fieldwork_regime
2024-Q1  2024_h1_macroeconomic_shock
2024-Q2  2024_h1_macroeconomic_shock
```

These rows have `ordinary_seasonality_eligible=false` and `ordinary_structural_time_eligible=false`. Downstream models may fit dedicated shock effects, so the observed dips remain reproducible while ordinary time/seasonality estimation can exclude the exceptional quarters. For 2024, ordinary year-level structure can therefore be learned from non-exceptional quarters.

## Release payload

A successful release contains:

```text
persons.csv
coverage.csv
schema_inventory.csv
monetary_lineage.csv
panel_links.csv
panel_households.csv
parent_lock.json
harmonization.json
qa.json
manifest.json
checksums.sha256
```

The release ID is content-addressed by the parent lock, harmonization policy, conversion manifest and chosen monetary reference period. Existing release directories are immutable.

## Offline fixture proof

`tests/test_longitudinal_frame.py` constructs all 37 synthetic quarterly parents. The fixture includes optional schema transitions (`CONDACT`, then `PP04A`), zero income, repeated-wave household/person candidates, a deliberate demographic conflict, all three exceptional quarters and an approved monetary conversion surface. It also verifies failure on a missing quarter, an unsupported required-schema transition and an unapproved conversion release.

## L2 local packet after merge

With actual parents already pinned under `/home/matias/data/eph-releases/`:

```bash
python scripts/17_build_longitudinal_parent_ledger.py \
  --pinned-root /home/matias/data/eph-releases \
  --output /home/matias/data/eph-longitudinal-2017-2026/parent-lock.json

python scripts/18_build_longitudinal_analysis_frame.py \
  --parent-lock /home/matias/data/eph-longitudinal-2017-2026/parent-lock.json \
  --pinned-root /home/matias/data/eph-releases \
  --monetary-conversion /path/to/immutable/ipc-conversion-release \
  --monetary-reference-period YYYY-MM-01 \
  --output-root /home/matias/data/eph-longitudinal-2017-2026/releases
```

L2 must then inspect `schema_inventory.csv`, `panel_links.csv`, `monetary_lineage.csv`, `coverage.csv` and `qa.json`. Any real historical transition not represented by the current conservative rules remains unresolved until source evidence justifies an explicit rule. The final longitudinal welfare model is out of scope here.
