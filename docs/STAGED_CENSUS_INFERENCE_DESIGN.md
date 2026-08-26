# Staged Census income inference design

> **Boundary status (2026-08-26): provisional bridge.** The staged design and its
> out-of-fold training rule remain valuable research evidence, but the presence of
> `census_income.py` and Census packaging helpers in this repository does not make
> this thesis/experiment workspace the long-term owner of execution over Census
> sample releases. Future promotion should preserve the scientific design while
> allowing scoring orchestration to live with the exact Census sample consumer.

## Decision

The Census path is a separate, explicit staged estimator. It is **not** a mode of
the frozen EPH flagship estimator.

The reusable primitive is `StagedClassifierRegressor`. A configuration adapter
can declare any bounded sequence of stages using `StageSpec`:

```text
Census-observable columns
  -> classifier wave 1 -> predicted categorical feature
  -> classifier wave 2 -> predicted categorical feature
  -> classifier wave 3 -> predicted categorical feature
  -> income regressor
```

Every stage names its target, output feature, and required inputs. Later stages
may list earlier predicted outputs among their inputs. This represents the legacy
successive-classifier design without copying orchestration code for every target.

## Training rule

Downstream training uses **out-of-fold predictions**, not the observed classifier
targets. Each classifier is subsequently refitted on the complete training frame
for inference. This distinction matters because otherwise the intermediate
targets would leak into the final income regressor and produce unrealistically
optimistic training evidence.

The synthetic smoke fixture demonstrates three classifier waves (`activity`,
`occupation`, and `education`) followed by an income regressor. It exists only to
exercise contracts and has no Census or poverty scientific claim.

## Release gate

A real release is allowed only after all of the following are supplied and
verified:

1. one immutable `research.census-sample/v1` input release;
2. the exact Census-observable feature definition for every stage;
3. historical fitted model binaries or approved training inputs;
4. a declared output transform from code or manifest evidence;
5. stable, exact one-to-one person-period coverage;
6. model, input, output, and manifest hashes;
7. an explicit monetary-reference status.

The packager rejects fuzzy/positional ID matches, duplicates, missing or extra
IDs, non-finite values, and contradictory transforms. Negative finite values on a
log scale are warnings rather than failures. Monetary reference may remain
`unresolved`, but that status is visible in the manifest and cannot be silently
promoted.

## Current local result

The governed inventory is recorded in
`reports/census_income/artifact_inventory.json`. No local Census release, RFC4
output, RFC1–RFC3 family, fitted historical classifier, or sibling historical
workspace was present. Accordingly, no inference was attempted and no real
`research.person-income-predictions/v1` release was emitted. This is acceptance
path 3 (precisely blocked), not a substitution with the flagship HGB model.

The next recovery action is to mount or copy the historical
`indice-pobreza-UBA` and `encuestador-de-hogares` workspaces together with the
Census release, RFC files, fitted models, rank tables, and run metadata, then run:

```bash
make census-inference-check
```

If a canonical, exactly matched prediction table is recovered, package it with:

```bash
make census-income-release \
  CENSUS_RELEASE=/copied/census-release \
  PREDICTIONS=/recovered/canonical-person-predictions.csv \
  OUTPUT_ROOT=/local/releases/income/<immutable-release-id> \
  MONETARY_STATUS=unresolved
```

Then validate the immutable payload:

```bash
make census-income-release-check RELEASE_DIR=/local/releases/income/<immutable-release-id>
```
