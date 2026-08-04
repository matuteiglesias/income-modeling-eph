# Legacy preprocessing migration

Git history commit `61357b5` records four 100%-similarity Git renames. This is direct repository evidence that the
legacy artifact name meant an annual analysis input, not a train split.

| Period | Former name | Current name | Rows × columns | Classification |
|---|---|---|---:|---|
| 2022 | `data/training/EPHARG_train_22.csv` | `data/annual_preprocessed_inputs/EPHARG_annual_input_22.csv` | 189,581 × 56 | byte-identical rename |
| 2023 | `data/training/EPHARG_train_23.csv` | `data/annual_preprocessed_inputs/EPHARG_annual_input_23.csv` | 199,766 × 56 | byte-identical rename |
| 2024 | `data/training/EPHARG_train_24.csv` | `data/annual_preprocessed_inputs/EPHARG_annual_input_24.csv` | 198,604 × 62 | byte-identical rename |
| 2025 | `data/training/EPHARG_train_25.csv` | `data/annual_preprocessed_inputs/EPHARG_annual_input_25.csv` | 47,950 × 62 | byte-identical rename |

There are no column renames, type changes, transformation changes, or sample-membership changes in this local
migration: Git's `R100` records unchanged file content. The current SHA-256 values are recorded in the release
manifests. The filenames alone changed.

This proves local artifact continuity, not independent equivalence with an external checkout of
`encuestador-de-hogares`. That repository and its `crear_EPH_training` implementation could not be retrieved in the
execution environment. Producer-version identity, upstream raw inputs, and external duplicate copies remain
unresolved pending read-only upstream reconciliation. Legacy student scripts in this repository reference precisely
the four former `EPHARG_train_<yy>.csv` names, corroborating their role as modeling inputs.
