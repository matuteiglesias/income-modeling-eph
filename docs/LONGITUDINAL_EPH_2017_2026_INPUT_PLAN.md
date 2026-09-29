# Longitudinal EPH input surface, 2017-Q1 to 2026-Q1 — implementation plan

Status: implementation-ready design, 2026-09-29.

## Mission

Create the governed EPH-side longitudinal evidence required by the new welfare-transport study without turning `encuestador-de-hogares` into an EPH acquisition/preprocessing owner.

The immediate window is:

```text
2017-Q1 .. 2026-Q1
```

This repo consumes exact `publicdata.eph-microdata@1` releases from `microdatos-EPH-INDEC` and produces a neutral longitudinal analysis/input surface. It does not score Census rows.

## Why this is a new lane

The existing flagship inputs cover 2022-Q1..2025-Q1 and were built for the Spisso-motivated EPH-only modeling study. They should remain frozen evidence.

The longitudinal welfare study needs:

- all available quarters 2017-Q1..2026-Q1;
- stable person and household identity including period;
- zero-income persons retained where valid;
- source-faithful labor state for EPH-side diagnostics;
- no target-derived geography ranks;
- explicit monetary conversion rather than hidden legacy normalization;
- household-safe repeated-wave grouping;
- a COVID-2020-Q2 policy.

Do not mutate the flagship dataset in place.

## Parent acquisition

`microdatos-EPH-INDEC` remains raw/source custody authority.

Before analysis-frame construction, build a 37-quarter parent ledger with:

```text
period
source_release_id
source_manifest_sha256
individual_table_sha256
household_table_sha256
row counts
schema fingerprint
status
```

The producer already supports quarterly source releases. Add consumer-side discovery/verification here only as needed; do not copy raw acquisition logic.

Fail closed on a missing/invalid parent. Do not silently substitute a nearby quarter.

## Target longitudinal analysis artifact

Conceptually:

```text
research.eph-longitudinal-analysis-frame/v1
```

Minimum person-level semantics:

```text
row_id
household_observation_id
panel_household_id
person_key_within_household
ANO4
TRIMESTRE
period
AGLOMERADO
region_id
P47T_nominal
P47T_real
monetary_reference_period
source_release_id
source_row_identity
CONDACT / ESTADO source-faithful labor fields
approved shared / derived-shared candidate features
survey weights preserved with explicit semantic names
exception_flags
```

Do not call a row globally longitudinal unless identity linkage is proven.

## Repeated-wave identity

INDEC documents the EPH 2-2-2 rotation and `CODUSU` supports following dwellings through quarters. The current encuestador runner already groups repeated household waves together.

Build and audit a panel-link surface separately from ordinary row identity.

At minimum distinguish:

```text
household observation identity
panel dwelling/household identity
person linkage candidate
person linkage confidence/status
elapsed_quarters
```

Do not assume `COMPONENTE` is permanently stable across all waves without an explicit audit.

The useful observed EPH panel horizon is about 1.5 years; it does not directly identify 2010→2026 persistence.

## Monetary normalization

The new longitudinal amount target must use an exact `IPC-Argentina` conversion release.

Preserve both:

```text
P47T_nominal
P47T_real
```

Choose and freeze one common real reference period for model training. Record:

- conversion release ID;
- source period;
- target reference period;
- factor;
- round-trip QA.

Deflation occurs before fitting the positive-income amount head. The zero/positive hurdle state is not changed by nominal deflation.

Do not reuse the historical January-2016 normalization unless it is independently re-authorized through the current monetary contract.

## 2020-Q2 policy

Keep 2020-Q2 in the artifact and preserve the actual COVID-era observation.

Mark it explicitly:

```text
pandemic_fieldwork_regime
```

Downstream default structural fitting policy:

- exclude 2020-Q2 from estimation of ordinary recurring quarter seasonality;
- exclude it from any "normal-period" structural sensitivity where requested;
- permit a dedicated 2020-Q2 period/shock correction so an actual 2020-Q2 measurement can still be produced;
- always report a with/without-2020-Q2 sensitivity.

Do not delete the quarter.

## Feature boundary

This frame is neutral evidence for transport research, not the EPH-only flagship feature contract.

Must preserve enough source fields to build:

1. the deployable common EPH/Census plane;
2. EPH-only richer information-ceiling diagnostics;
3. labor-transition/panel diagnostics.

It must not materialize `AGLO_rk` or `Reg_rk` as accepted deployment features because they are target-derived.

## Cloud work packet — C2

Implement the longitudinal consumer/build surface.

Deliver:

1. 37-quarter parent discovery/lock format;
2. schema-drift inventory and explicit harmonization rules;
3. source-faithful person/household join with period-qualified IDs;
4. repeated-wave/panel audit tooling;
5. IPC conversion hook + exact monetary lineage;
6. 2020-Q2 exception policy;
7. longitudinal artifact manifest/checksums/coverage;
8. fixture tests spanning schema transitions;
9. no changes to the frozen 2022-2025 flagship artifacts.

Prefer extending the already-proven neutral analysis-frame path rather than reactivating legacy annual CSV machinery.

## Local work packet — L2

After C2:

1. materialize/verify all 37 exact EPH parent releases;
2. build the real longitudinal frame;
3. produce a schema-transition report;
4. audit repeated household/person linkage;
5. produce nominal↔real monetary QA;
6. verify period coverage and row accounting;
7. isolate any quarter that cannot satisfy the stable feature contract.

Suggested external roots:

```text
/home/matias/data/eph-releases/
/home/matias/data/eph-longitudinal-2017-2026/
```

## Definition of done

A single immutable longitudinal EPH analysis release exists for 2017-Q1..2026-Q1, with exact parent lineage, common-real-income target, household-safe identities and explicit 2020-Q2 status, and can be consumed by `encuestador-de-hogares` as an artifact rather than by importing this repo runtime.

## Non-goals

- no Census scoring;
- no poverty classification;
- no forecast/nowcast;
- no decision yet that one pooled model is scientifically promoted;
- no overwrite of Spisso/flagship evidence.
