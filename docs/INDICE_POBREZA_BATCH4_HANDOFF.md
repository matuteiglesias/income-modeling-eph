# Batch 4 consumer handoff

The candidate model release predicts `log10(P47T)` on the `log10_ars` scale. It does **not**
predict `log10(P47T + 1)`, and consumers must not add or remove one.

The model was trained and validated on EPH-derived person rows. It does not create predictions
for a Census sample and is explicitly incompatible with Census-mode inference. The downstream
poverty system may consume only a separate immutable person-income prediction release whose
sample-ID namespace matches its Census sample. Producing that release requires a separately
approved Census inference/adaptation packet.

`indice-pobreza-UBA` must not load this model binary directly under this packet. A synthetic
income-prediction fixture may support adapter and pure-kernel development, but it carries no
scientific Census or poverty-measurement claim. A mechanical `10**prediction` is not an unbiased
expected income because retransformation bias remains unresolved.
