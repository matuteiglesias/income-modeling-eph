# L2 — real 2017-Q1..2026-Q1 longitudinal EPH commissioning gate

Status: local-data execution packet, 2026-09-29.

## Purpose

Execute the real-data gate after merged C2.

Authoritative architecture:

- `docs/LONGITUDINAL_EPH_2017_2026_INPUT_PLAN.md`
- `docs/LONGITUDINAL_EPH_2017_2026_IMPLEMENTATION.md`

C2 implemented `research.eph-longitudinal-analysis-frame/v1`. L2 must materialize the real 37-quarter evidence surface, adjudicate actual historical schema behavior and freeze the monetary timing choice used by downstream longitudinal welfare measurement.

This is a **local/data and scientific-adjudication task**.

## Required parent window

```text
2017-Q1 .. 2026-Q1
37 exact publicdata.eph-microdata@1 parents
```

Raw/source custody remains in `matuteiglesias/microdatos-EPH-INDEC`.

If one or more exact quarterly parents do not yet exist locally, materialize them through that producer. Do not synthesize a parent inside this repo.

Suggested roots:

```text
/home/matias/data/eph-releases/
/home/matias/data/eph-longitudinal-2017-2026/
```

## Gate A — parent and schema census

1. Materialize/verify all 37 exact upstream releases.
2. Build the exact C2 parent ledger.
3. Inspect actual person and household schemas by quarter.
4. Produce a schema-transition census.
5. For every required-field transition, either:
   - prove identity/compatibility from source evidence and add an explicit governed harmonization rule; or
   - isolate the affected quarter and stop downstream commissioning.
6. No guessed aliases.

## Gate B — identity and repeated-wave audit

Inspect:

- `panel_links.csv`;
- `panel_households.csv`;
- demographic-conflict counts;
- elapsed-quarter distribution;
- 1- and 3-quarter repeat support;
- irregular/attrition gaps.

Quantify how often `CODUSU + NRO_HOGAR + COMPONENTE` behaves as a credible short-horizon person-link candidate.

Do not promote it to permanent identity.

This receipt becomes the empirical support boundary for L11/L12.

## Gate C — monetary timing adjudication

C2 currently supports deterministic quarter-to-month conversion and the implementation default maps:

```text
Q1 -> February
Q2 -> May
Q3 -> August
Q4 -> November
```

Do **not** freeze that convention merely because it is implemented.

Before the real longitudinal release is scientifically frozen:

1. inspect the official EPH income reference-period/questionnaire semantics available for the historical window;
2. determine the highest-resolution monetary timing actually recoverable from the source;
3. compare the implemented midpoint convention against at least one source-justified alternative, such as a quarter-average conversion or interview/reference-month-resolved conversion if the source supports it;
4. quantify the effect on real `P47T`, year effects and downstream aggregate levels on a bounded sample/period set;
5. select and document one rule for the centerline;
6. retain the alternatives as sensitivity policies if material.

The selected monetary rule must be deterministic, versioned and bound to an immutable `IPC-Argentina` conversion release.

No silent January-2016 fallback.

## Gate D — exceptional periods

Verify that:

```text
2020-Q2  pandemic_fieldwork_regime
2024-Q1  2024_h1_macroeconomic_shock
2024-Q2  2024_h1_macroeconomic_shock
```

remain present as observed rows, with ordinary seasonality/structural-time eligibility disabled.

Do not remove these observations.

## Gate E — final release

Build the immutable real longitudinal frame under the adjudicated monetary rule.

Required checks:

- 37/37 periods;
- exact parent hashes;
- row accounting;
- no orphan person rows;
- zero-income retention;
- nominal/real round-trip QA;
- schema inventory accepted;
- panel audit persisted;
- exception metadata correct;
- release checksum verification.

## Receipt

Persist:

```text
release_id
release_path
manifest_sha256
37-quarter parent ledger identity
schema-transition report
panel-link audit
monetary timing decision
monetary sensitivity summary
IPC conversion release identity
exception-period verification
coverage / row accounting
validation commands and results
```

## Definition of done

One immutable real `research.eph-longitudinal-analysis-frame/v1` release exists for 2017-Q1..2026-Q1 and is suitable as the observed EPH parent for the longitudinal semantic plane and L10 commissioning.

## Non-goals

- no Census scoring;
- no L10/L11/L12 model promotion;
- no poverty estimate;
- no forecast/nowcast.
