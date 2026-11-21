# DARC v0.5 Myopic i0 on Qwen3.8 Full-8

> The existing 140-task Qwen3.8 cohort is split outcome-blind into 112 development tasks and 28 retrospective held-out tasks. The separate 20-task prospective cohort remains untouched for later confirmation.

## Protocol

- Controller: unchanged DARC v0.5 Myopic iteration-0 targets.
- Development: 112 Qwen3.8 Full-8 tasks; four grouped OOF folds; three seeds.
- Evaluation: 28 outcome-blind task-ID held-out tasks from the existing 140.
- Future confirmation: 20 sealed prospective tasks, not accessed here.
- Online decisions never observe future artifacts, outcomes, or realized future call cost.
- Frozen development reference budget: **14,932.8 tokens/task**.
- OOF-selected model variant before held-out evaluation: **scratch**.

## Retrospective held-out full-budget results

| Policy | Accuracy mean | Actual tokens | Calls | Helpful / Harmful / No-change |
|---|---:|---:|---:|---:|
| Always VVV | 67.86% | 398,698 | 84.0 | 13.0 / 0.0 / 71.0 |
| DARC v0.5 Myopic i0 · Qwen3.8 scratch | 67.86% | 330,383 | 68.3 | 13.0 / 0.0 / 55.3 |
| Always CP3 | 64.29% | 149,871 | 28.0 | 12.0 / 0.0 / 16.0 |
| DARC v0.5 Myopic i0 · Qwen2.5 transfer | 60.71% | 316,004 | 65.0 | 11.0 / 0.0 / 54.0 |
| Always CP1 | 60.71% | 133,685 | 28.0 | 11.0 / 0.0 / 17.0 |
| Always CP2 | 60.71% | 130,506 | 28.0 | 11.0 / 0.0 / 17.0 |
| Never Verify | 21.43% | 0 | 0.0 | 0.0 / 0.0 / 0.0 |

![Held-out accuracy–cost frontier](../figures/heldout_accuracy_cost.png)

## Interpretation

This 28-task result is a retrospective internal holdout because the 140-task cohort had already been inspected previously. It is useful for immediate model development but is not final confirmatory evidence. The untouched 20-task prospective cohort remains the later confirmation set.

## Artifacts

- `split_integrity.json` and `tables/task_folds.csv`
- `checkpoints/<variant>/seed_<seed>/iteration_0.pt`
- `tables/oof_results.csv`, `tables/heldout_results.csv`
- `tables/policy_grid.csv`, `tables/decision_trace.csv`
