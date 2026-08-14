# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A hockey analytics project that trains a PyTorch neural network to predict expected goals (xG) for NHL shots, using historical shot-event data. There is no package structure, build system, dependency manifest, or test suite — this is a small collection of standalone, run-top-to-bottom research scripts intended to be executed locally (Mac) or submitted as batch jobs to a Compute Canada Slurm cluster ("Narval").

## Repo layout

- `nn_xgoals.py` — baseline training script (full-batch, CPU, local file paths active).
- `nn_xgoals_gpu.py` — same pipeline, moves model/tensors to CUDA if available (`torch.device`), Narval file paths active.
- `nn_xgoals_batching.py` — same pipeline but uses `DataLoader`/`TensorDataset` mini-batch training instead of full-batch.
- `code_for_cluster/nn_xgoals.py` — exact duplicate of the top-level `nn_xgoals.py`, kept alongside `code_for_cluster/xgoals.sh` so the whole cluster submission is self-contained in one directory.
- `xgoals.sh`, `xgoals_gpu.sh` — Slurm `sbatch` submission scripts (CPU and GPU) for Narval.
- `example_hockey_script.sh` — a generic/annotated Slurm template (not wired to any script here); also contains a `gtime -v` benchmark transcript of running `nn_xgoals.py` locally as a reference for expected runtime/memory.
- `cluster_plots/`, `gpu_plots/` — checked-in example output plots from prior runs of the CPU and GPU scripts respectively.

These three `nn_xgoals*.py` scripts are near-duplicates of one pipeline, not separate modules that import from each other — when changing modeling logic (features, architecture, training loop), check whether the same change should be mirrored across all of them (including `code_for_cluster/nn_xgoals.py`).

## Running the scripts

No `requirements.txt`/`pyproject.toml` exists; dependencies are installed inline by the Slurm scripts. To run locally:

```bash
pip install numpy matplotlib pandas seaborn scikit-learn scipy torch
python3 nn_xgoals.py
```

To submit to the Narval cluster:

```bash
sbatch xgoals.sh       # CPU
sbatch xgoals_gpu.sh   # GPU
```

These scripts `module load python/3.13.2 scipy-stack`, build a throwaway virtualenv in `$SLURM_TMPDIR`, `pip install --no-index` the offline wheels, then run the target `.py` file. **Known issue:** `xgoals.sh` invokes `python nn_goals.py`, which does not match any file in this repo (the real script is `nn_xgoals.py`) — fix this filename before relying on that submission script.

There is no lint config, formatter config, or test suite in this repo.

## Environment-specific paths (edit before running)

Each script hardcodes two absolute paths near the top, under `# LOAD DATA`, with the inactive environment's values commented out directly above/below:

- `shots_dir` — directory of per-season shot CSVs (`shots_<year>_<year+1>.csv`), external data not checked into this repo.
- `plot_folder` — output directory for generated plots.

`nn_xgoals.py` and `nn_xgoals_batching.py` currently have the local Mac paths active; `nn_xgoals_gpu.py` has the Narval cluster paths active. When moving a script between machines, comment/uncomment these two variables rather than introducing a config file.

## Input data schema

Each `shots_<season>.csv` must contain at minimum: `goal` (label), `xGoal` (existing external model's prediction, used for comparison), `shooterName`, `isPlayoffGame`, `lastEventCategory`, and the feature columns listed under `features` in each script (shot angle/distance/type/rebound info, time-on-ice fields, `offWing`, `shotOnEmptyNet`, etc.). Rows are filtered to regular-season shots and dead-play `lastEventCategory` values (`PENL`, `STOP`, `GOAL`, `CHL`, `PEND`, `PSTR`, `ANTHEM`, `EISTR`, `GEND`, `EGT`) are dropped before feature selection.

## Pipeline shared by all three `nn_xgoals*.py` scripts

1. **Load & filter**: concatenate one CSV per training season, filter out playoff games and dead-play events; load a separate held-out season CSV (`data_predict`) for final evaluation the same way.
2. **Feature selection**: fixed `features` list (see script) split into numeric vs. categorical (`object` dtype) columns.
3. **Preprocess**: `ColumnTransformer` one-hot encodes categorical features (`remainder="passthrough"` keeps numeric/binary features as-is); the same transformer is then re-fit separately on `X` and `X_predict` (not fit once and reused) — this is `fit_transform` called independently on each, so keep that in mind if changing this section, since it means train/predict one-hot columns aren't guaranteed identical before the correlation/variance filtering step.
4. **Feature pruning**: drop one of each pair of features with Pearson correlation > 0.9, then drop low-variance features (`VarianceThreshold(threshold=0.005)`).
5. **Split**: 80/20 train/val split, stratified on `goal`, `random_state=1997`.
6. **Model** (`XGoalNeuralNetwork`, defined per-script): 4 hidden fully-connected layers (512→256→128→64) each with BatchNorm1d + LeakyReLU (ReLU on the last hidden layer) + Dropout(0.3), single logit output. Loss is `BCEWithLogitsLoss`; optimizer is Adam (`weight_decay=1e-4`).
7. **Training loop**: manual epoch loop (mini-batched via `DataLoader` only in `nn_xgoals_batching.py`) with a 20-epoch linear LR warmup followed by `StepLR` decay, and early stopping on validation loss (`patience=30`). `torch.manual_seed(1997)` is set before model init. The best `state_dict` (lowest val loss) is reloaded before evaluation.
8. **Evaluation & plots**: saves loss curves, ROC curve, PR curve, and calibration curve to `plot_folder`, then applies the trained model to `data_predict`, producing a histogram of predicted shot probabilities, a per-shooter table of actual vs. predicted (summed) goals, an actual-vs-predicted regression plot with a 95% CI band, and distribution-comparison plots between the model's xG and the pre-existing `xGoal` column already present in the data.

`nn_xgoals_batching.py` currently has `epochs = 5` (a debug value — full runs elsewhere use `epochs = 1000`) and has the ROC/PR curve block wrapped in `if False:` (disabled), so don't assume its plot output matches the other two scripts without checking.
