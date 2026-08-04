# Preprocessing decisions required

No item below is silently resolved by Batch 1. Review is required before a new value-producing annual release.

1. **Source releases:** replace `artifact:publicdata.eph-microdata@1-provisional` with per-period immutable release
   IDs and hashes after reconciliation with `microdatos-EPH-INDEC`.
2. **Money:** identify the price index, series vintage, formula, base/reference period and variables adjusted. Decide
   how already-adjusted source variables are detected. Current historical values must not be re-deflated casually.
3. **Schema vintages:** approve the handling of 2024/2025 disaggregated `V2_*` and `V5_*` fields and document any
   upstream category/domain changes, especially special numeric missing codes.
4. **Geography:** identify the Region assignment source and whether a versioned crosswalk/geography release is
   actually consumed. Do not declare `eph-censo-aligner` as a dependency until code consumes its artifact.
5. **Ranks:** recover the exact population, income measure, grouping, tie policy and period used for `AGLO_rk` and
   `Reg_rk`; decide whether to freeze, retire, or reconstruct them. They remain leakage-suspect.
6. **Merges:** recover household/person source filenames, full keys and intended cardinalities. Decide how unmatched
   people/households and the observed complete duplicates are handled. A true person sequence identifier is needed.
7. **Indicators and filters:** recover formulas for `INGRESO*`, upstream exclusion timing, and missing-category
   behavior. The downstream `INGRESO == 1`, positive `P47T`, nonmissing `PROP` contract is unchanged.
8. **Legacy differences:** inspect the former authority read-only and compare its output hashes. Do not label any
   external difference intentional until reviewed.

## Batch 2 readiness

Batch 2 can consume the four characterized historical releases through
`configs/annual_input_consumer_contract.yaml` and require manifest schema `1.0`. It must preserve the warnings and
must not claim source-to-artifact reproducibility. Freezing a newly produced flagship release is blocked on source
hashes, monetary policy and substantive lineage review.
