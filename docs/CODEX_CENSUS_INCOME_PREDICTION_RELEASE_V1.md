# Codex work packet — Census-indexed person-income prediction release v1

## Mission

Produce the first immutable `research.person-income-predictions/v1` release whose
rows are indexed in the exact namespace of a `research.census-sample/v1` release.

This is the missing bridge between the Census sample and the poverty kernel.

Do not apply the frozen EPH flagship HGB model directly to Census rows. The
flagship remains an EPH benchmark and evidence release. This task must recover,
characterize and bound the historical Census-inference path or package an existing
matching final prediction output.

The fastest valid result is preferred over unnecessary retraining.

## Read first

Inspect:

- `docs/INDICE_POBREZA_BATCH4_HANDOFF.md`;
- the frozen flagship release documentation and inference contract;
- annual preprocessing/model authority declarations;
- current feature and target contracts;
- the historical poverty consumer audit;
- `encuestador-de-hogares` read-only as former model/preprocessing lineage;
- the historical four-stage prediction code and variable lists in
  `indice-pobreza-UBA` read-only;
- local ignored model/output paths only when available.

Do not change the flagship model, target, split or benchmark in this task.

# Part A — inventory before inference

Search read-only for historical artifacts that may already solve the immediate
problem:

```text
RFC4_<fraction>_<period>_<scope>.csv
RFC1/RFC2/RFC3 families used to construct RFC4
fitted_RF/ or modelos/ classifier files
matching Census sample/population tables
feature/rank tables
run logs or notebooks recording transform and monetary reference
```

Likely historical locations include repository-local ignored paths and, when they
exist:

```text
/media/matias/Elements/suite/out/
/media/matias/Elements/suite/poblaciones/
data/Fitted_RF/
data/Pobreza/
```

Do not assume those paths exist. Discover from the current machine and repository
history.

Create an inventory with:

- absolute local path, but never embed it in a release contract;
- SHA-256 and byte size;
- row count and columns;
- period and sample fraction;
- person-ID evidence;
- transform evidence;
- monetary-reference evidence;
- model/stage identity;
- compatibility with each discovered Census sample;
- whether the artifact can be packaged without recomputation.

# Part B — prefer packaging a matching final output

If an existing final RFC4/person-income output shares the exact person IDs of a
Census sample release, package it before attempting new inference.

The candidate must prove:

- exact one-to-one coverage of the sample person IDs;
- no duplicate person-period keys;
- selected period agreement;
- transform identified from code, manifest or strong artifact evidence;
- monetary reference identified or truthfully declared unresolved/candidate;
- model lineage and source file hash recorded;
- finite values;
- deterministic output ordering.

Do not guess `log10_ars` versus `log10_ars_plus_1` from a filename. Inspect the
actual legacy code path and value semantics.

When evidence is incomplete but the artifact is otherwise usable, publish a
`legacy_candidate` with warnings rather than silently filling provenance.

# Part C — recover the historical Census-inference path only if needed

If no matching final output exists, reconstruct the smallest viable historical
inference path.

The old poverty workspace attempted a staged system resembling:

```text
Census-observable variables
  -> labor/activity status stages
  -> intermediate socioeconomic outcomes
  -> final person income
```

Characterize each stage before running it:

- model binary and serialization format;
- software/library compatibility;
- target and output domain;
- required features;
- Census-observable versus EPH-only features;
- generated/intermediate features;
- missing-category handling;
- rank and employment adjustments;
- period and monetary assumptions;
- whether the stage is actually required for final income.

Do not migrate all historical notebook code. Build a bounded inference adapter
around verified artifacts.

## C1. No flagship substitution

The frozen HGB model predicts `log10(P47T)` on EPH-derived persons. It must remain
explicitly incompatible with Census-mode inference unless a separate scientific
study demonstrates otherwise.

Never make a Census prediction by simply aligning similarly named columns and
loading the flagship estimator.

## C2. Stop conditions

Stop the inference path rather than fabricate output when:

- a required historical model is absent;
- model deserialization cannot be made reproducible;
- a required feature has no evidenced Census definition;
- person IDs cannot be kept stable;
- the stage output transform is ambiguous;
- a hidden rank/employment artifact materially changes prediction and cannot be
  identified.

A stop should produce a precise compatibility report and missing-artifact list.

# Part D — input and output contracts

Consume exactly one copied immutable Census sample release:

```text
research.census-sample/v1
```

Produce:

```text
research.person-income-predictions/v1
```

Canonical columns:

```text
sample_person_id
period
prediction_value
prediction_transform
monetary_reference
classification
model_release_id
```

Allowed transform vocabulary:

```text
linear_ars
log10_ars
log10_ars_plus_1
```

Expected classification for modeled Census income:

```text
projected
```

The manifest must declare:

- exact Census release ID and manifest hash;
- sample-ID namespace;
- period;
- prediction transform;
- currency and monetary reference;
- model/stage release identities;
- source hashes;
- feature-contract identity;
- output file hash and row count;
- QA, warnings and limitations;
- upstream EPH/model lineage when applicable.

# Part E — align the monetary reference

The first poverty slice uses a basket product tied to:

```text
research.argentina-price-composite/legacy-compatible-v1@2016-01=100
```

Do not silently relabel income as sharing that reference.

Determine whether the historical prediction output already uses the compatible
reference. Evidence may come from:

- the training/preprocessing code;
- the historical IPC file consumed;
- a retained model/input manifest;
- exact code-lineage correspondence.

Classify the conclusion:

```text
verified
probable_by_code_lineage
unresolved
incompatible
```

A `probable_by_code_lineage` candidate may proceed with a prominent warning when
Matías accepts that candidate status. An incompatible reference requires an
explicit deterministic conversion artifact; do not perform an implicit scalar
conversion in the poverty consumer.

# Part F — warning-oriented range QA

The objective is iteration and visibility, not rejecting every unusual model
output.

Hard failures:

- NaN or infinity;
- checksum/identity mismatch;
- duplicate or missing person IDs under the selected strict release;
- unknown or contradictory transform;
- double transformation;
- output row count different from the exact Census sample without a declared
  exclusion policy;
- nondeterministic output from the same inputs and models.

Warnings:

- finite negative predictions on a log scale;
- very low or high predicted linear income;
- distribution shift relative to EPH training data;
- unresolved retransformation bias;
- probable rather than verified monetary reference;
- legacy model/library compatibility;
- partial or bounded geographic scope.

Do not clip merely to make plots attractive. Report the distribution and make
candidate status explicit.

# Part G — QA and visible diagnostics

Produce deterministic QA tables and plots:

- prediction distribution on native and linear scales;
- quantiles and extreme finite values;
- predictions by region and department;
- prediction coverage by Census sample IDs;
- transformed-versus-native consistency checks;
- demographic summaries by age and sex;
- comparison with EPH validation predictions only as a domain-shift diagnostic,
  not as proof of Census accuracy;
- warning counts and affected rows.

Create a human-readable handoff report for `indice-pobreza-UBA` containing the
release path, hashes, namespace, period, transform, monetary reference and exact
limitations.

# Part H — fixture and contract tests

Create a synthetic Census fixture and simple deterministic staged-model fixture
covering:

- exact ID preservation;
- multiple households and departments;
- categorical features;
- missing categories;
- finite negative log predictions;
- duplicate/missing/extra IDs;
- stage-output schema mismatch;
- transform mismatch;
- monetary-reference warning;
- deterministic rerun.

Provide commands equivalent to:

```bash
make census-inference-check
make census-inference-smoke
make census-income-release \
  CENSUS_RELEASE=/copied/census-release \
  OUTPUT_ROOT=/local/releases/income
make census-income-release-check RELEASE_DIR=...
```

# Non-goals

- No flagship model tuning or retraining.
- No claim that EPH validation metrics measure Census prediction quality.
- No poverty calculation.
- No Census sampling.
- No fuzzy or positional ID matching.
- No automatic monetary-reference approval.
- No large Census/prediction artifact committed to Git.
- No scheduled cross-repository execution.

# Acceptance paths

## Path 1 — existing matched predictions

```text
matching Census sample and RFC4/final output found
-> identities and transform proved
-> immutable legacy-candidate release emitted
-> QA and handoff produced
```

## Path 2 — bounded historical inference

```text
models and feature contracts found
-> deterministic inference adapter implemented
-> one candidate prediction release emitted
-> QA and handoff produced
```

## Path 3 — precisely blocked

```text
required model/feature artifact absent or ambiguous
-> no predictions fabricated
-> compatibility and missing-artifact report emitted
-> exact next human/local recovery step stated
```

# Completion report

The final PR must state:

- local artifacts and repositories inspected;
- whether an existing matching output was found;
- exact Census sample release consumed;
- inference/package path chosen;
- model/stage identities and hashes;
- prediction release ID, path and manifest hash;
- period, namespace, transform and monetary-reference status;
- row coverage and distribution diagnostics;
- warnings versus hard failures;
- commands and tests run;
- downstream handoff command for `indice-pobreza-UBA`;
- confirmation that the EPH flagship model was not silently applied to Census.
