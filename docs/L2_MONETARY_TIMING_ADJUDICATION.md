# L2 monetary timing adjudication

The EPH parent files provide quarterly observations and `P47T`, but do not
carry a respondent-level reference month. The centerline therefore maps each
quarter to its fieldwork midpoint: Q1→February, Q2→May, Q3→August, and
Q4→November. This is a transparent quarter-level proxy for the monthly
income reference and is recorded in `monetary_lineage.csv`; it is not treated
as an observed month.

The conversion parent is `arg-monetary-conversion-v1-d1bc61d6ddb94c81` with
common reference `2026-01-01` and producer base `2016-01-01`. The materialized
L2 run used the candidate conversion release in bounded commissioning mode;
approved-mode consumers must still require an approved conversion manifest.

For sensitivity, the same source-faithful frame can be rematerialized with
quarter-start (Q1/Q2/Q3/Q4 → January/April/July/October), quarter-end
(March/June/September/December), or a geometric quarter-average factor. Across
the 37 quarters, start versus midpoint factors differ by 4.18% on average in
absolute value, end versus midpoint by 4.42%, with a 20.15% maximum at
2023-Q4. These are conversion-scale sensitivities, not alternative source
observations; no interpolation is applied to EPH income.
