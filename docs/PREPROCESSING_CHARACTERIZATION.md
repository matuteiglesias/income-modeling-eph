# Current preprocessing characterization

## Status and authority

`income-modeling-eph` is the current authority for
`artifact:research.eph-annual-preprocessed@1`. The four committed CSVs are historical,
already-materialized outputs. No checked-in function can reproduce them from raw EPH files today. The executable
local path therefore **validates and releases** these files; it does not pretend to recreate their values.

Facts below are either **code-observed** (current files/code/history) or **historical intent** (legacy names and
comments). This distinction is deliberate.

## Executable paths

1. `scripts/11_preprocessing_authority.py` reads each annual CSV, checks every column against
   `configs/annual_input_lineage.yaml`, calculates content/schema hashes, coverage, duplicate diagnostics and
   missingness, and writes or validates release manifests.
2. `scripts/01_build_dataset.py` loads the four annual inputs in configured order; drops `V2_01_M`, `V2_02_M`,
   `V2_03_M`, `V5_01_M`, `V5_02_M`, and `V5_03_M` from 2024/2025; concatenates; performs modeling-stage feature
   engineering; filters `INGRESO == 1`, `P47T > 0`, and nonmissing `PROP`; constructs `log10(P47T)`; and excludes
   forbidden predictors.
3. The synthetic fixture executes a bounded many-person-to-one-household merge, geography mapping, explicit
   monetary factor, stubbed rank boundary, unexpected/missing geography, invalid merge, and sample inclusion.

The former producer path `crear_EPH_training` is not present locally and network inspection of
`matuteiglesias/encuestador-de-hogares` was unavailable during this characterization. Consequently, raw archive,
household and person filenames, exact merge keys, crosswalk implementation, price series, and exclusion order are
unresolved rather than guessed.

## Materialized annual releases

| Current file | Coverage | Rows | Columns | SHA-256 |
|---|---:|---:|---:|---|
| `EPHARG_annual_input_22.csv` | 2022 Q1–Q4 | 189,581 | 56 | `05397f7e7c7ce174ffba4e17bcbbdfbbc5790d3e9aaebfb94b3a056908fc2dd3` |
| `EPHARG_annual_input_23.csv` | 2023 Q1–Q4 | 199,766 | 56 | `1788a3d9829e7772829e9a99adafc51e2ad5c77406c3ef74b2f5f413a56ece4a` |
| `EPHARG_annual_input_24.csv` | 2024 Q1–Q4 | 198,604 | 62 | `b8262193afd1a495f31b45d7fd72e795259829b5a1bc7084fa7741c5d4661d1f` |
| `EPHARG_annual_input_25.csv` | 2025 Q1 only | 47,950 | 62 | `460a6ee7b64eda71745fce2d3f069af903d1f747aa7b015079477aebe122d1e0` |

The manifests are the complete source for column inventories, inferred pandas types and per-column missingness.
2024/2025 add six disaggregated `V2_*`/`V5_*` monetary columns. All other names are shared. Types differ by vintage
where missing values force numeric columns from integer to floating representation; three text columns occur in
each file.

## Observed lineage and transformations

The authoritative registry is `configs/annual_input_lineage.yaml`. It lists all 62 observed columns, entity level,
source variable, producer attribution, transformation class, units, domain/missingness status, supported vintages,
consumer uses, leakage sensitivity, legacy status, and review status.

Code-observed facts are:

* `CODUSU` is the only retained household identifier. A true person sequence identifier is absent. The candidate
  diagnostic key (`CODUSU`, `ANO4`, `TRIMESTRE`, `P02`, `P03`) is not unique, and complete duplicate rows exist
  (225, 238, 243 and 68 by release). No rows are silently removed.
* `Region`, `AGLOMERADO`, `AGLO_rk`, and `Reg_rk` are materialized. Rank lookup tables exist at
  `data/info/AGLO_rk` and `data/info/Reg_rk`; current annual ranks are accepted, not recomputed.
* Rank columns are target-derived/leakage-suspect. They are optional for the clean consumer contract.
* Monetary variables are already materialized in the annual files. Their price series and reference period are not
  recorded in executable local code, so the manifests use explicit provisional/unresolved identifiers.
* `INGRESO`, `INGRESO_NLB`, `INGRESO_JUB`, and `INGRESO_SBS` are materialized indicators. Only the precisely
  configured modeling filter is executable locally; their upstream formulas are unresolved.
* Missing values are retained. Unexpected category values are not normalized at the annual release boundary.
  Modeling-only recodes and derived sanitation, household age composition, education maximum, and log income
  components occur later in `src/eph_income/features.py` and do not mutate annual artifacts.

## Merge and source boundary

Historical intent says the former authority merged household and person records and assigned geography before
writing `EPHARG_train*`. Exact upstream filenames, keys and cardinality exception handling are not preserved here.
The release contract now requires a stable household key, an explicit person key, and a many-person-to-one-household
validation before a future source-backed release can publish. Until reconstructed, manifests warn that candidate
key uniqueness is diagnostic only.

The provisional source release is `artifact:publicdata.eph-microdata@1-provisional`. No Census crosswalk dependency
is declared because current code does not consume one. Reconciliation requirements are in the decision log.

## Mutation and publication

Ordinary `make validate`, `make preprocessing-smoke`, and `make preprocessing-release-fixture` do not write annual
inputs. `make preprocessing-manifests` is the explicit manifest-writing command. Publication is blocked until
source hashes, monetary reference, lineage review, and validations are complete; current manifests characterize
historical files and carry warnings rather than manufacturing evidence.
