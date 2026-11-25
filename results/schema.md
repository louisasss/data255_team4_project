# Results schema

One CSV per run, written by `src/runner.py` to `results/runs/<run_id>.csv` (one header line, one row). Column order is fixed as listed below.

## run_id format

`{bands}_b{budget}_s{seed}`, e.g. `rgb_b10_s1`, `all_b250_s3`.
Defined in `src/experiment_grid.py`. 

**Note:** this does not yet include the init strategy or pretraining source, so ablation and SSL4EO runs would collide with main-comparison runs. Pending a team decision.

## Columns

| Column | Type | Description |
|---|---|---|
| run_id | str | Run identifier (see format above) |
| model | str | Backbone, currently always `resnet18` |
| bands | str | `rgb`, `rgb_nir`, or `all` |
| budget | int / empty | Images per class (10, 50, 250); empty = full training set |
| seed | int | Seed for both subset selection and training (1, 2, 3) |
| eval_split | str | Split the metrics are computed on; `val` until Week 6 |
| best_val_epoch | int | Epoch (0-indexed) with the highest validation accuracy; ties keep the earlier epoch |
| accuracy | float | Overall accuracy at best_val_epoch (primary metric) |
| macro_f1 | float | Macro-averaged F1 at best_val_epoch |
| recall_<Class> | float | Per-class recall, one column for each of the 10 classes, in `src.metrics.CLASSES` order |
| val_loss | float | Average validation cross-entropy at best_val_epoch |
| train_time_s | float | Wall-clock seconds for the whole training loop, **including** per-epoch validation |
| params | int | Total parameter count (`src.models.count_parameters`) |
| device | str | `mps`, `cuda`, or `cpu` |
| checkpoint | str | Path to the saved best-validation weights (not committed to git) |

## Rules

- Metrics are always taken from the best-validation epoch, never the last epoch.
- The test set is not evaluated until Week 6. Test metrics will be added as
  separate rows with `eval_split = test`, using the saved checkpoint.
- Column names match `results/baseline_val.csv` where they overlap, so
  `src/compare_results.py` can read both.
- Results from stub or smoke-test runs are never written to `results/`.