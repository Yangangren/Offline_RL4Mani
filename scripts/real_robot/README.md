# Real-robot RGB Diffusion Policy pipelines

This folder contains the offline real-robot pipelines for the pick-cup and
stack-cup tasks. They convert the recorded packages, validate the results,
prepare standard robomimic Diffusion Policy configs, and launch training. ROS
and robot-control deployment are deliberately out of scope.

## Quick start

Run these commands from the repository root with the robomimic environment:

```bash
# Revalidate the published shards and regenerate the production config.
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py --stages validate prepare

# Train the production baseline (250 epochs by default).
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py --stages prepare train
```

To exercise the original 20 Hz model and checkpoint path without starting a
production run:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py \
  --stages prepare train --smoke
```

The converted 20 Hz dataset already lives in `datasets/real_robot/pick_cup`. To
build it again from the source package, use `--stages dataset prepare`; existing
valid shards are reused only when their conversion settings match the request.
Add `--force-dataset` only when intentionally replacing both shards.

## Stack-cup 20 Hz baseline

The stack-cup source is
`/home/ryan/datasets/stack_cup/human_demo`. The converter preserves normalized
seven-dimensional commands at 20 Hz and maps each action row to the latest
paired 5 Hz RGB capture available at that time. It never applies the physical
action scale. Episode 007 is excluded because its internal camera gap would
produce a 0.6175-second-old observation, beyond the fixed 0.5-second contract.

Build the single committed HDF5 shard, validate it against the raw source, and
prepare the fingerprinted production config:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_stack_cup_rgb_dp_baseline.py \
  --stages dataset validate prepare
```

The fixed masks contain 49 usable episodes: 44 in `train` and episodes
004/024/027/040/046 in `valid`. The same shard provides `train_clean`, a
32-episode strict subset whose source QA passed and whose precomputed model
windows contain no invalid entries.

Launch the default expanded-data baseline. The epoch count is written
explicitly here even though 250 is also the launcher default:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_stack_cup_rgb_dp_baseline.py \
  --stages validate prepare train \
  --train-mask train \
  --epochs 250
```

For the strict-data comparison, use `--train-mask train_clean`; the mask is
included in the experiment name so it cannot overwrite the expanded run. A
two-update GPU smoke run is available with `--smoke --name-suffix preflight`.

Evaluate a trained checkpoint on every held-out window with the online network
and the rollout-time EMA network reported separately:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/eval_stack_cup_rgb_dp.py \
  --checkpoint /absolute/path/to/model_epoch_250.pth \
  --seeds 1 2 3 \
  --batch-size 64 \
  --device auto \
  --output /absolute/path/to/stack_cup_epoch250_heldout_eval.json
```

EMA replay compares all eight predicted action slots and reports normalized
motion MAE/RMSE, physical translation and rotation RMSE, and gripper sign
accuracy per slot. These remain open-loop imitation diagnostics, not robot
success measurements.

## Round-1 5 Hz baseline

The deployment-oriented 5 Hz variant is separate from the original 20 Hz,
two-round baseline. It uses the 49 QA-eligible demonstrations from collection
round 1 (source episodes 002--050), a fixed four-source-command grid, and one
fresh policy action per 5 Hz observation. The production dataset is
`datasets/real_robot/pick_cup_5hz_round1/round1_5hz_rgb.hdf5`.

Validate the published dataset, regenerate its fingerprinted config, and train:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_5hz_round1.py \
  --stages validate prepare train
```

To rebuild the 5 Hz shard from the raw package before training, include the
`dataset` stage. A matching existing shard is validated and reused; pass
`--force-dataset` only when intentionally replacing it.

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_5hz_round1.py \
  --stages dataset validate prepare train
```

The 5 Hz observation/action/prediction horizons are `2/1/4`. Each normalized
motion target is the componentwise mean of four 20 Hz source commands and is
decoded over 0.2 seconds with scales 0.048 m and 0.144 rad. The gripper target
is the dense logical state after the fourth source command. A robot adapter must
interpolate and rate-limit this macro target; it must not apply the maximum
translation or rotation as an instantaneous jump.

To exercise the 5 Hz model and checkpoint path without starting a production
run:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_5hz_round1.py \
  --stages validate prepare train --smoke --name-suffix preflight
```

### Held-out checkpoint evaluation

`validate` in the launcher checks the HDF5 data contract; it does not load a
trained policy. Use the checkpoint evaluator to test all held-out windows. It
reports the online-network diffusion validation loss separately from EMA
rollout-action replay, uses fixed seeds, does not shuffle, and does not drop the
last partial batch.

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/eval_pick_cup_rgb_dp_5hz_round1.py \
  --checkpoint /absolute/path/to/model_epoch_100.pth \
  --seeds 1 2 3 \
  --batch-size 64 \
  --device auto \
  --output /absolute/path/to/model_epoch_100_heldout_eval.json
```

The command evaluates the five `valid`-mask demonstrations only. The action
errors are open-loop command-imitation metrics, not closed-loop task success.
Use `--max-windows N` only for a quick smoke test.

## 20 Hz mixed-data chunk IDQL

The real pick-cup chunk-IDQL task is exposed as `pick_cup` in the repository
launcher. Raw deployment rollouts are read from
`/home/ryan/datasets/pick_cup/rollout/{success,failure}` and converted to
`datasets/real_robot/pick_cup/idql/pick_cup_epoch200_20hz_rollouts.hdf5`.
The converter preserves 20 Hz state and normalized seven-dimensional policy
actions while sampling RGB on a causal wall-clock 5 Hz grid. Images normally
repeat for about four action rows; after a blocking gripper or control pause,
the next row advances to the latest causal grid tick instead of retaining a
stale pre-pause frame.

One held-out successful rollout has a documented recorder-startup gap. Its
first seven source actions are trimmed as one disconnected prefix (rather than
using a future frame or relaxing the 0.5 s image-age contract); this does not
change any fitting-set count. Every retained rollout row remains causal.

The deterministic fitting split contains exactly 99 episodes:

- 65 human demonstrations: 44 `train` episodes from round 1 and 21 from round
  2. The two source `valid` masks (five episodes per round) remain excluded.
- 23 of 29 successful rollouts, with the other six in `success_valid`.
- 11 of 14 failed rollouts, with the other three in `failure_valid`.

The mixed fitting output is
`datasets/real_robot/pick_cup/idql/pick_cup_chunk_idql_65demo_23success_11failure_terminal_success.hdf5`.
It uses `terminal_success`: a successful episode has its sole reward of 1 on
the final recorded transition, while a failure has zero reward throughout;
both terminate at the recorded episode end. The default actor condition is
`human_only`, so human rows have condition 1 and every rollout row has
condition 0. The first training run uses `pretrained_dp_joint`, initializing
from the deployed epoch-200 Diffusion Policy and optimizing the actor jointly
with Q/V; the actor is not frozen.

From the repository root, build the converted rollout and mixed fitting data:

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup build_dataset
```

Running the same stage again without `OVERWRITE_DATASET=1` is the launcher
validation path: it checks raw-source provenance, deterministic masks, source
identities, schema, reward terminals, action/observation shapes, and the mixed
dataset contract without replacing either output.

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup build_dataset
```

Only after that command succeeds, launch the default joint-actor run:

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup train_chunk_idql
```

The build and training stages also validate existing inputs before use. Their
successful terminal output is the source of truth for full-corpus conversion
validation; this documentation does not imply that a particular local build
has already passed. The generic `eval_chunk_grid_resilient`,
`collect_chunk_idql_rollouts_resilient`, and composed simulation stages are
intentionally rejected for `pick_cup`: they target robomimic simulation and
must not be used as a real-robot execution client.

## Data contract

- Two HDF5 shards preserve collection rounds 1 and 2. The standard robomimic
  `MetaDataset` gives each round equal total sampling mass.
- Observations are paired main and wrist RGB images at 96x128, EEF position,
  EEF quaternion in `xyzw` order, and the logical gripper state before the
  current action.
- Actions are six already-normalized Cartesian motion channels plus a dense
  post-action gripper target (`-1` closed, `+1` open).
- Image selection is causal against the actual camera header timestamps, not
  nominal frame times. A sample is rejected if its selected pair is more than
  0.5 seconds old.
- Raw gripper events, source row indices, timestamps, selected frame indices,
  camera stamps, and image ages are retained under each demo's `provenance`
  group.
- Spatially diverse validation masks are deterministic and disjoint from the
  training masks.

## Implementation map

- `build_pick_cup_dataset.py`: source audit, conversion, split creation, and
  rollback-safe two-shard publication guarded by a generation commit marker.
- `validate_pick_cup_dataset.py`: independent schema, timing, provenance,
  gripper, mask, and cross-shard validation.
- `run_pick_cup_rgb_dp_baseline.py`: config generation, standard-loader
  preflight, balanced multi-shard sampling, and training launch.
- `pick_cup_common.py`: shared schema and source-contract helpers.

The baseline reuses `robomimic/algo/diffusion_policy.py` and the existing
robomimic dataset loader. Its default horizons are observation/action/prediction
`2/8/16`, it uses both cameras with 84x112 random crops, DDIM with 10 inference
steps, EMA, and the full `[256, 512, 1024]` temporal U-Net.

## Verification

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -m unittest \
  tests.real_robot.test_build_pick_cup_dataset \
  tests.real_robot.test_pick_cup_rgb_dp_baseline \
  tests.real_robot.test_pick_cup_5hz_pipeline \
  tests.test_train_utils_validation_scheduler -v
```

The converter also writes `datasets/real_robot/pick_cup/conversion_summary.json`
with episode counts, sample counts, causal prefix drops, and maximum image ages.
`dataset_commit.json` is the publication marker; the launcher rejects missing or
mixed shard generations. Training configs also fingerprint each shard, so an old
checkpoint is never silently reused after data or hyperparameters change.
