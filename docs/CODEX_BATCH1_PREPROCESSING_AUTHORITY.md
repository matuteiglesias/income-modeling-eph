# Codex work packet — Batch 1: establish the EPH preprocessing authority

## Mission

Make this repository the explicit, trustworthy authority for the annual preprocessed EPH artifacts used by its modeling system.

The owner decision is already made:

> `income-modeling-eph` is the current preprocessing authority. `encuestador-de-hogares` is the former authority and a source of lineage evidence. The legacy artifacts called `EPHARG_train*` are the predecessor of the current, less confusingly named `EPHARG_annual_input_*` artifacts.

Your task is to prove, document, test, and package that authority without changing the scientific target or model behavior.

## Why this matters

The modeling layer is already comparatively strong: it has explicit feature and leakage contracts, guarded execution, diagnostic profiles, experiment planning, and thesis evidence gates. Its weak seam is the identity and reproducibility of the annual preprocessed inputs.

Batch 2 can connect and freeze the flagship model only after those inputs become versioned releases with column lineage, source identities, manifests, and compatibility evidence.

## Read first

1. Read every applicable `AGENTS.md` file.
2. Read `README.md`, `SYSTEM.yaml`, `Makefile`, package/environment files, `scripts/01_build_dataset.py`, preprocessing helpers, tests, configuration files, and the contents/metadata of `data/annual_preprocessed_inputs/` without loading large files unnecessarily.
3. Read `configs/feature_contract.yaml`, split configuration, run/evidence registries, and data-boundary documentation.
4. Inspect `matuteiglesias/encuestador-de-hogares` read-only for the former `crear_EPH_training` path, `EPHARG_train*` outputs, geographic ranks, deflation, household/person merging, and schema patches.
5. Inspect `matuteiglesias/indice-pobreza-UBA` read-only for downstream assumptions about the preprocessed artifacts.

Do not make cross-repository changes from this packet.

## Authority and boundaries

This repository owns:

- acquisition-to-analysis preprocessing after receiving an explicit EPH source release;
- household and individual merges used by the modeling dataset;
- variable harmonization used by this research program;
- region and agglomerate assignment used in the annual inputs;
- monetary normalization/deflation used in those inputs;
- derived geographic ranks and sample indicators used in those inputs;
- annual preprocessed artifact schema and releases;
- model targets, features, leakage exclusions, splits, training, metrics, and evidence.

It does not own:

- official EPH source publication;
- raw archive acquisition and DBF conversion;
- official Census or geographic authority;
- poverty-threshold methodology;
- official income or poverty statistics.

## Required deliverables

### 1. Current preprocessing characterization

Create `docs/PREPROCESSING_CHARACTERIZATION.md` describing:

- the exact code path that creates or validates each `EPHARG_annual_input_<yy>.csv` file;
- every upstream file expected;
- household/person merge keys and cardinality assumptions;
- variable harmonization and schema normalization;
- region/agglomerate assignment;
- monetary normalization and reference period;
- geographic rank construction;
- sample and training indicators;
- duplicate, missingness, and exclusion policies;
- all columns added, removed, renamed, or transformed;
- which steps are currently executable and which rely on already-materialized files.

Distinguish code-observed facts from historical intent.

### 2. Correct the declared boundary

After characterization, update `README.md` and `SYSTEM.yaml` so they no longer describe an unspecified external preprocessing repository as the current owner.

Declare a versioned artifact identity such as:

```text
artifact:research.eph-annual-preprocessed@1
```

Choose the final identifier consistently with the estate conventions. Record:

- the raw/source release consumed;
- optional crosswalk/geography releases consumed;
- the annual preprocessed release produced;
- the modeling dataset and model releases produced downstream.

Do not claim a dependency that is not actually used.

### 3. Legacy-to-current equivalence map

Create `docs/LEGACY_PREPROCESSING_MIGRATION.md` mapping:

```text
encuestador-de-hogares: EPHARG_train*
    → income-modeling-eph: EPHARG_annual_input_*
```

For each available legacy/current pair, record:

- file identity and period;
- row/column counts;
- schema differences;
- renamed columns;
- type differences;
- transformation differences;
- sample-membership differences;
- whether the pair is byte-identical, value-equivalent after normalization, semantically equivalent, intentionally changed, or unresolved.

Do not use “same dataset” as a substitute for evidence. The owner decision establishes continuity of authority, not byte identity.

### 4. Column-level lineage contract

Create a machine-readable lineage registry for every column in the annual inputs, including:

- column name and type;
- entity level: household, person, geography, time, or derived;
- source variable(s);
- producing step/function;
- transformation class;
- unit and monetary reference where relevant;
- allowed missingness/domain;
- first and last supported vintage;
- whether used by target, feature, split, filter, diagnostics, or provenance only;
- leakage sensitivity;
- legacy name/status;
- reviewer status for substantive transformations.

Generated documentation may be derived from the registry, but the machine-readable form is authoritative.

### 5. Annual-input release manifest

For every annual artifact, emit a manifest containing at least:

- manifest schema version;
- release ID;
- source-release IDs and hashes;
- preprocessing code commit;
- configuration and lineage-registry hashes;
- year/quarter coverage;
- row/column counts;
- column inventory and schema hash;
- file size and SHA-256;
- entity/key uniqueness checks;
- missingness summary;
- monetary reference and price-series identity;
- geography/crosswalk identity when used;
- warnings and unresolved lineage;
- producing command and environment.

A release manifest must not embed local absolute paths.

### 6. Bounded preprocessing fixture

Create a small synthetic fixture representing:

- multiple households and persons;
- a valid and invalid merge case;
- at least two periods/years;
- regional/agglomerate mapping;
- monetary adjustment;
- geographic ranks or their explicitly stubbed boundary;
- missing and unexpected categories;
- sample inclusion/exclusion;
- columns relevant to the feature contract and leakage exclusions.

The fixture should exercise preprocessing only. It must not require full model training.

### 7. Deterministic preprocessing checks

Expose canonical commands equivalent to:

```bash
make validate
make test
make preprocessing-smoke
make preprocessing-release-fixture
make preprocessing-manifests
```

Tests must verify:

- merge cardinalities and stable keys;
- deterministic schema and row ordering;
- no target leakage introduced by preprocessing;
- explicit monetary reference;
- stable category and type normalization;
- manifest consistency with files;
- no mutation of committed annual inputs during ordinary checks;
- failure before publishing incomplete releases.

Keep full model and expensive experiment execution behind existing guards.

### 8. Compatibility report for consumers

Create a deterministic consumer contract containing:

- required columns for dataset building;
- columns used by `feature_contract.yaml`;
- columns used for target/filter/split construction;
- deprecated legacy names;
- minimum supported annual-input manifest version;
- expected behavior when optional crosswalk or rank columns are absent.

This contract is the input to Batch 2 and to the downstream audit in `indice-pobreza-UBA`.

### 9. Reproducibility decision log

Create `docs/PREPROCESSING_DECISIONS_REQUIRED.md` listing any unresolved human decisions, including:

- monetary index and base/reference policy;
- treatment of schema changes across EPH vintages;
- geographic rank construction;
- household/person merge exceptions;
- sample filters;
- intentional differences from the legacy producer.

Do not silently choose a new scientific policy.

## Ordered execution

1. Characterize the current preprocessing and materialized inputs.
2. Characterize the legacy producer read-only.
3. Build the legacy/current equivalence map.
4. Establish the column-lineage registry.
5. Add the bounded fixture and characterization tests.
6. Add release manifests and deterministic validation.
7. Establish the consumer contract.
8. Correct README and `SYSTEM.yaml` to reflect current authority.
9. Run existing cheap modeling checks only after preprocessing changes are stable.
10. Produce the decision log and Batch 2 readiness statement.

## Coordination with the other Batch 1 packets

This packet may begin immediately, but final source and crosswalk release identifiers may remain provisional until:

- `microdatos-EPH-INDEC` establishes the EPH source-release contract;
- `eph-censo-aligner` establishes its mapping release, if actually consumed;
- `censo-ign-geografias` establishes its geography release, if actually consumed.

Do not block local lineage work on those packets. Use explicit provisional IDs and list the reconciliation needed.

## Human checkpoints

Stop for review before:

- changing any value-producing transformation;
- changing sample membership or merge cardinality policy;
- changing monetary normalization;
- replacing or recomputing geographic ranks;
- changing target or leakage exclusions;
- deleting legacy/current annual artifacts;
- running expensive full training;
- declaring unresolved legacy/current differences intentional.

## Non-goals

- No new model family or hyperparameter search.
- No thesis-result reinterpretation.
- No full poverty pipeline.
- No raw EPH downloader implementation.
- No wholesale rewrite of working experiment infrastructure.
- No automatic regeneration or commit of all annual inputs.
- No large data added to Git.
- No changes to `encuestador-de-hogares` or `indice-pobreza-UBA` from this branch.

## Stop conditions

Stop rather than guess when:

- an annual input cannot be tied to a producer code path;
- a legacy/current difference affects values or sample membership without evidence;
- the monetary reference cannot be established;
- merge keys are not unique as assumed;
- a column's meaning changes across vintages;
- completing the task would require an unapproved methodological change.

## Acceptance criteria

```text
income-modeling-eph is truthfully declared as current preprocessing authority
legacy EPHARG_train and current annual-input artifacts have an evidence-backed migration map
all current annual-input columns have machine-readable lineage or are explicitly unresolved
a bounded preprocessing fixture passes deterministic tests
annual-input manifests include sources, code, schema, coverage, counts, units, and hashes
consumer requirements are explicit
ordinary checks do not mutate annual inputs or run expensive training
human methodological decisions remain visible
Batch 2 can select an annual-input release without reconstructing hidden lineage
```

## Completion report

The final response and PR description must include:

- exact files, commands, and tests changed;
- the identified current preprocessing entry points;
- legacy/current artifact pairs compared and their equivalence classifications;
- lineage coverage and unresolved columns;
- fixture and manifest outputs;
- corrections made to repository authority declarations;
- decisions still required from Matías;
- an explicit statement of whether the repository is ready for Batch 2 model connection.
