# Codex work packet — Portfolio Batch 2: freeze the flagship income model

## Mission

Freeze one existing scientific result as an immutable, consumer-verifiable model release whose complete path is:

```text
four characterized annual EPH artifacts
    → locked modeling dataset and split assignments
    → fixed feature and experiment contracts
    → one canonical guarded experiment
    → diagnostics and evidence gates
    → one candidate model release
```

The objective is **not a better model**. It is the same controlled scientific result, now attached to defensible artifact identity, execution lineage, limitations, and an inference boundary.

## Governing selection

Use the current experiment registry as the authority.

The initial flagship candidate is:

```text
hgb_quick_benchmark_v1
```

because the registry labels it `main_hgb_benchmark`, it uses the `clean_geo` feature view, and its fixed configuration excludes target-derived `AGLO_rk` and `Reg_rk`.

Supporting evidence may reference:

```text
baseline_income_prediction_v1
hgb_quick_clean_geo_v1
ols_core
ols_core_aglo_plus_time_fe
```

but do not rerun those supporting experiments merely to complete this packet. Do not change the flagship model, feature view, hyperparameters, split, target, or metric without Matías's approval.

## Read first

1. Read every applicable `AGENTS.md` file.
2. Read `README.md`, `SYSTEM.yaml`, `Makefile`, `EXPERIMENTS.md`, the experiment registry, the selected experiment config, feature contract, preprocessing consumer contract, annual-input manifests, dataset builder, experiment runner, guards, artifact writers, diagnostics/evidence scripts, and tests.
3. Inventory existing `data/processed/`, `reports/runs/`, experiment cards, convenience copies, and thesis evidence by metadata and hashes before running anything expensive.
4. Treat current annual-input manifests as historical artifact attestations: they identify materialized files but explicitly do not prove reproduction from raw EPH.
5. Preserve the target exactly as:

```text
logP47T = log10(P47T)
```

This is **not** `log10(P47T + 1)`. Never silently translate between those contracts.

## Authority and limitations

This repository owns:

- annual preprocessed EPH artifact attestation and consumer validation;
- modeling-dataset construction;
- split assignments;
- target, feature, leakage, experiment, diagnostics, and evidence contracts;
- training and candidate model releases;
- inference against an explicitly compatible feature table.

This repository currently does not prove:

- raw EPH-to-annual-input reproduction;
- the historical price series used to materialize annual files;
- a true unique person key in the annual source artifacts;
- compatibility with a Census sample;
- an official or causal income estimate;
- suitability of a mechanically exponentiated prediction for poverty measurement.

A model release produced here must therefore start at status `candidate` and carry these limitations.

# Phase A — converge the annual-input integration contract

## A1. Shared artifact envelope

Upgrade each annual-input manifest to retain all current preprocessing-specific fields while also exposing the shared envelope:

```text
research-artifact-manifest/v1
```

Required common fields:

- manifest schema;
- artifact type `research.eph-annual-preprocessed`;
- release ID;
- release status;
- producer repository;
- manifest-attestation commit;
- deterministic creation timestamp derived from that commit;
- method/contract version;
- data vintage;
- inputs;
- files with roles, sizes, and SHA-256 hashes;
- reports with roles, sizes, and hashes;
- limitations.

Use `candidate` for the four historical annual releases. Do not mark them `reviewed` or `approved` automatically.

## A2. Do not falsify producer identity

Separate these concepts explicitly:

```text
artifact_materialization_commit
manifest_attestation_commit
```

The current code can identify the commit that generated the manifest, but the historical commit that originally materialized the annual CSV may be unresolved. Record it as unresolved instead of using the current commit for both meanings.

The manifest generator must not retain an obsolete base commit merely to make `--check` pass. A regenerated manifest must be bound truthfully to the attestation code/config used, while deterministic checks must define which fields are expected to change across commits.

## A3. Compatibility declaration

Add a machine-readable compatibility declaration for:

```text
research.eph-annual-preprocessed/v1
    → research.eph-model-training/v1
```

It must declare:

- supported manifest schema versions;
- required years and files;
- minimum required columns;
- vintage-specific schema handling;
- target and inclusion contract;
- optional geography/rank behavior;
- approved-mode policy;
- unresolved source and monetary-reference policy;
- no Census-sample compatibility.

## A4. Standard-library preflight validator

Add a lightweight validator that runs before pandas or model code. It must validate:

- manifest and compatibility schema versions;
- manifest checksum when supplied;
- safe repository-relative paths;
- file existence, size, and SHA-256;
- artifact type and release status;
- requested vintage coverage;
- input/source-manifest identities;
- lineage registry, config, consumer-contract, and schema hashes;
- unresolved source and monetary fields under candidate versus approved mode;
- all four annual artifacts required by the model lock.

Negative tests must cover tampering, stale hashes, missing files, unsupported schema, unknown status, wrong vintage, absolute/path-escaping files, and approved-mode rejection.

# Phase B — create the immutable model-input lock

Create a model-input lock whose identity is content-derived from stable inputs, not timestamps or local paths.

The lock must pin:

- the four annual manifest identities and hashes;
- the four annual CSV hashes;
- annual lineage-registry hash;
- preprocessing consumer-contract hash;
- preprocessing release configuration hash;
- dataset-builder code/contract version;
- feature-contract hash;
- selected experiment-config hash;
- experiment-registry entry hash or normalized identity;
- processed modeling-dataset hash;
- dataset metadata hash;
- split-assignments hash;
- split strategy and random seed;
- target contract;
- feature view;
- environment identity;
- repository commit.

Provide commands equivalent to:

```bash
make model-freeze-preflight
make model-input-lock
make model-input-lock-check
```

Ordinary validation must not rebuild annual files or silently overwrite an existing lock.

If the processed dataset or split assignments do not exist, construct them through the canonical dataset builder only after all annual-input preflight checks pass.

# Phase C — resolve or reproduce the canonical run

## C1. Inventory before training

Search existing run directories for a run matching all of:

- experiment ID `hgb_quick_benchmark_v1`;
- selected config hash;
- feature-contract hash;
- clean-geo feature view;
- processed-dataset hash;
- split-assignments hash;
- random seed;
- expected model parameters;
- complete predictions, metrics, diagnostics, card, and manifest artifacts.

Create `docs/FLAGSHIP_RUN_RESOLUTION.md` recording:

- candidate runs found;
- exact compatibility checks;
- selected run or reason no run can be reused;
- missing/stale artifacts;
- whether retraining is required.

Do not choose “latest” by directory name alone.

## C2. Exact rerun only when needed

If no existing run matches the lock, run exactly:

```bash
make run-hgb-quick-benchmark
```

using the existing guard and fixed configuration. Do not tune, expand the grid, change the sample, or substitute another experiment.

If the execution environment cannot complete the guarded run, stop with:

- the exact successful preflight evidence;
- the exact command required;
- estimated resource surfaces observed from prior runs/config, without fabricating runtime;
- no candidate model release.

Do not automatically rerun the full multi-model baseline.

## C3. Diagnostics and evidence

For the selected run:

- build required diagnostics from saved predictions;
- run existing evidence collection/check gates;
- verify both test and validation outputs required by the registry;
- fail if convenience copies disagree with the canonical run directory;
- retain the run directory as canonical authority.

# Phase D — freeze one candidate model release

## D1. Release identity and envelope

Produce an immutable release with artifact type:

```text
research.eph-income-model/v1
```

and status:

```text
candidate
```

The manifest must use `research-artifact-manifest/v1` and include:

- model release ID;
- producer repository and commit;
- deterministic timestamp;
- input-lock identity and hash;
- experiment ID/config hash;
- target and feature contracts;
- selected model class and exact parameters;
- training/test/validation row counts;
- split identity;
- environment and dependency versions;
- metrics and diagnostics artifacts with hashes;
- prediction artifacts with hashes;
- training-frame sample identity;
- limitations and unresolved upstream lineage;
- compatibility declaration.

## D2. Fitted estimator boundary

Inspect whether the current runner persists the selected fitted pipeline.

If it does not, implement a narrowly scoped **freeze-only** persistence path for the selected best estimator. Do not make every ordinary experiment serialize every fitted candidate.

Requirements:

- persist the full fitted preprocessing/model pipeline;
- record serialization library and version;
- hash the binary;
- never load it before validating the release envelope and file hash;
- document that Python pickle/joblib artifacts are trusted-local artifacts and must never be loaded from an untrusted source;
- include an environment lock or exact dependency snapshot;
- retain a model-independent prediction fixture so compatibility can be checked without trusting arbitrary binaries.

## D3. Inference contract

Expose an inference interface against a table matching the trained EPH feature contract.

The contract must declare:

- required columns and domains;
- feature order after deterministic preparation;
- target output name;
- target scale `log10_ars` corresponding to `log10(P47T)`;
- no automatic `+1` correction;
- no Census-sample compatibility;
- behavior for missing/extra columns;
- output row identity;
- provenance fields.

Do not automatically publish an inverse-transformed ARS value as an unbiased expected income. If a mechanical `10**prediction` field is offered, label it explicitly as a mechanical inverse transformation and document retransformation bias. The primary model output remains on the declared target scale.

## D4. Release validator and smoke

Provide commands equivalent to:

```bash
make model-release
make model-release-check
make model-release-smoke
```

The smoke must:

- copy the immutable release outside the producer working tree;
- validate all hashes and compatibility metadata;
- score a bounded synthetic/approved fixture with the same feature contract;
- preserve row identity and ordering;
- reject a tampered model, config, input lock, or fixture schema;
- reject Census-mode use;
- never invoke sibling repositories.

# Phase E — Batch 4 handoff

Create a consumer handoff for `indice-pobreza-UBA` that states exactly:

- the model release predicts `log10(P47T)`, not `log10(P47T + 1)`;
- the release is trained and validated on EPH-derived rows;
- it does not by itself create predictions for a Census sample;
- the downstream poverty system may consume only a separate immutable person-income prediction release whose sample-ID namespace matches its Census sample;
- producing such a release requires a separately approved Census inference/adaptation packet;
- a synthetic income-prediction fixture may be used to develop poverty adapters and the pure kernel without making a scientific claim.

Do not make `indice-pobreza-UBA` load the model binary directly as part of this packet.

# Required tests

At minimum add tests for:

- shared annual manifest envelope;
- materialization versus attestation commit semantics;
- full annual-manifest validation;
- candidate versus approved-mode behavior;
- deterministic model-input lock;
- stale processed dataset or split rejection;
- run selection by identity rather than recency;
- selected-run artifact completeness;
- deterministic release manifest;
- tampered release rejection;
- inference fixture schema and row identity;
- explicit `log10_ars` target metadata;
- Census incompatibility rejection;
- no sibling working-tree dependency.

# Human checkpoints

Stop for Matías before:

- replacing `hgb_quick_benchmark_v1` as the flagship candidate;
- changing any target, inclusion, split, feature, leakage, or model parameter;
- rerunning a different or larger experiment;
- promoting status beyond `candidate`;
- treating unresolved source/monetary lineage as resolved;
- claiming Census compatibility;
- using inverse-transformed predictions for poverty measurement;
- publishing model binaries or data artifacts externally.

# Non-goals

- No new models or hyperparameter search.
- No temporal or grouped split redesign.
- No fold-safe geography encoding project.
- No raw EPH reproduction.
- No price/basket repair.
- No Census sampling or Census prediction run.
- No poverty computation.
- No dashboard or deployment service.
- No deletion of historical runs.
- No automatic promotion to reviewed/approved.

# Acceptance criteria

```text
all four annual inputs validate through a shared candidate-release envelope
materialization and attestation provenance are not conflated
one deterministic model-input lock pins every data/config/code/split identity
an existing matching flagship run is reused or the exact fixed benchmark is reproduced
required diagnostics and evidence gates pass from saved run artifacts
one immutable candidate model release is emitted and independently validated
the fitted-pipeline boundary is safe, hashed, versioned, and fixture-tested
the target is explicitly log10(P47T) / log10_ars with no silent +1
Census-sample use is explicitly rejected
no scientific policy, model search, or upstream lineage is silently changed
```

# Completion report

The final Codex response and PR description must state:

- annual release and lock identities/hashes;
- existing runs inspected and run selected;
- whether training occurred and the exact command;
- selected model parameters and metrics without overstating interpretation;
- all generated model-release files and hashes;
- target/inference semantics;
- unresolved source, monetary, person-key, and Census-compatibility limitations;
- exact checks run and results;
- whether the release is ready for `candidate` consumption;
- confirmation that no tuning, poverty computation, Census inference, or approval promotion occurred.
