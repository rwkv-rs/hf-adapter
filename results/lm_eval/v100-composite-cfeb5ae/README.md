# V100 formal `lm_eval` composite

This compact release artifact records the final 48-unit
`lm_eval==0.4.9.1` validation decision for the 0.1B, 0.4B, and 1.5B models at
batch sizes 1 and 8.

The original formal run completed all 48 units. After the batch-regrouping
fix, only the six affected 1.5B PIQA, OpenBookQA, and WinoGrande batch-pair
units were rerun, following the explicit release decision not to repeat the
unaffected 42 units. `composite-provenance.json` records both source revisions
and the replacement set; `manifest.jsonl` retains command, model, dataset, and
result provenance for all 48 units.

`validation.json` reports 48 units, `status: passed`, and no metric-stability
failures. Samples, model weights, checkpoints, W&B data, and large logs are not
included.
