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
`datasets/real_robot/pick_cup/idql/pick_cup_chunk_idql_65demo_23success_11failure_terminal_success_human_success_condition.hdf5`.
The chunk launcher also builds the disjoint held-out file
`datasets/real_robot/pick_cup/idql/pick_cup_chunk_idql_validation_10demo_6success_3failure_terminal_success_human_success_condition.hdf5`.
It contains all 10 human validation episodes, six successful validation
rollouts, and three failed validation rollouts (7,964 windows total). Both
files retain source episode identities, and training aborts if any identity
appears in both.
It uses `terminal_success`: a successful episode has its sole reward of 1 on
the final recorded transition, while a failure has zero reward throughout;
both terminate at the recorded episode end. The default chunk-actor condition
is `human_success`: human and successful-rollout rows have condition 1, while
failed-rollout rows have condition 0. Each build or validation prints these
semantics and the positive/negative episode and transition counts. The first
training run uses `pretrained_dp_joint`, initializing
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

The default critic is `rise_temporal_v2`: Q and V each consume the full
two-frame actor observation history and score an eight-action chunk. After
every epoch, rank zero evaluates the online actor and Q/V losses over every
held-out window with a fixed RNG seed and no updates. Metrics are recorded
under `validation/*`, and the lowest held-out diffusion loss is preserved as
`best_validation.pt`. The RISE-v2 output directory has a
`_rise_temporal_v2` suffix so it cannot be confused with the completed legacy
critic run.

The build and training stages also validate existing inputs before use. Their
successful terminal output is the source of truth for full-corpus conversion
validation; this documentation does not imply that a particular local build
has already passed. The generic `eval_chunk_grid_resilient`,
`collect_chunk_idql_rollouts_resilient`, and composed simulation stages are
intentionally rejected for `pick_cup`: they target robomimic simulation and
must not be used as a real-robot execution client.

## 20 Hz mixed-data one-step IDQL

The one-step launcher also exposes `pick_cup`. It deliberately reuses the
same deterministic 99-episode fitting split and `terminal_success` rewards in
the original file without the `_human_success_condition` suffix. Its stored
actor condition remains `human_only`; the chunk launcher uses a separate file
so an existing HDF5 is never silently reinterpreted.
Before training, the launcher revalidates both the converted rollout source
and the mixed external-link dataset.

Validate or rebuild the data contract, then start the default 50-epoch run:

```bash
bash run_rgb_dp_idql.sh pick_cup build_dataset
bash run_rgb_dp_idql.sh pick_cup train_resilient
```

The run initializes its trainable diffusion actor from the deployed epoch-200
checkpoint and uses `robot0_gripper_state` as the critic's late-fusion key.
Outputs go to
`trained_models/real_robot/pick_cup_rgb_dp/idql/65demo_23success_11failure_terminal_success`.
The launcher's generic `eval`, `eval_grid_resilient`, and composed evaluation
stages are rejected for `pick_cup` because they instantiate robomimic
simulation rather than the guarded real-robot client.

## Stack-cup mixed-data IDQL

Both IDQL launchers expose the repaired rollout corpus as `stack_cup`. The raw
source is `/home/ryan/datasets/stack_cup/rollout/{success,failure}`. Conversion
requires all 50 finalized episodes, the deployed epoch-200 checkpoint identity,
600 normalized actions per source episode, all 3,750 digest-verified inference
inputs, and the exact 32-success / 18-failure outcome partition.

Camera reconstruction preserves the causal 5 Hz / 20 Hz contract. Logger
startup rows are trimmed only when an exact golden NPZ proves a rosbag startup
drop and every later row is within the 0.5-second age bound. The two audited
adjacent equal robot-state timestamps are accepted only in their exact episode
and edge; backward or any additional repeated timestamp is rejected. The
production rollout output has 29,826 retained transitions, including 26/6
success train/valid episodes and 14/4 failure train/valid episodes.

The mixed fitting file is
`datasets/real_robot/stack_cup/idql/stack_cup_chunk_idql_44demo_26success_14failure_terminal_success_human_success_condition.hdf5`.
It contains 84 episodes and 45,668 transitions: 21,793 from the 44 human
`train` episodes, 15,494 from 26 successful rollout-train episodes, and 8,381
from 14 failed rollout-train episodes. It uses external HDF5 links and virtual
shifted `next_obs`, so the image data are not copied into the small mixed file.
The chunk launcher additionally builds
`datasets/real_robot/stack_cup/idql/stack_cup_chunk_idql_validation_5demo_6success_4failure_terminal_success_human_success_condition.hdf5`,
containing the five held-out human episodes and the 6/4 held-out successful /
failed rollouts (8,594 windows). It is evaluated in full after every chunk
training epoch, with the best held-out actor checkpoint retained as
`best_validation.pt`. One-step IDQL remains unchanged.

Build or revalidate the rollout and mixed datasets through either launcher:

```bash
bash run_rgb_dp_chunk_idql.sh stack_cup build_dataset
bash run_rgb_dp_idql.sh stack_cup build_dataset
```

Start the default first chunked run (joint actor, not frozen) or the one-step
comparison. Both use the same selected episodes and terminal-success targets;
only chunk IDQL stores the `human_success` actor condition:

```bash
bash run_rgb_dp_chunk_idql.sh stack_cup train_chunk_idql
bash run_rgb_dp_idql.sh stack_cup train_resilient
```

Both initialize the actor from
`trained_models/real_robot/stack_cup_rgb_dp/stack_cup_rgb_dp_ddim_s1/20260822155238/models/model_epoch_200.pth`
and use `robot0_gripper_state` for critic late fusion. Generic simulation eval
and collection stages are rejected for this real-robot task.

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
  tests.test_train_utils_validation_scheduler -v
```

The converter also writes `datasets/real_robot/pick_cup/conversion_summary.json`
with episode counts, sample counts, causal prefix drops, and maximum image ages.
`dataset_commit.json` is the publication marker; the launcher rejects missing or
mixed shard generations. Training configs also fingerprint each shard, so an old
checkpoint is never silently reused after data or hyperparameters change.
