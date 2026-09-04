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
004/024/027/040/046 in `valid`. The current corpus is published at
`datasets/real_robot/stack_cup/stack_cup_rgb.hdf5`. It also provides
`train_clean`, a 24-episode strict subset whose source QA passed and whose
precomputed model windows contain no invalid entries. The launcher fingerprints
this replacement, so it will not reuse a completed run trained on the old file.

Launch the default expanded-data baseline. The epoch count is written
explicitly here even though 250 is also the launcher default. The canonical
experiment name is `stack_cup_rgb_dp_ddim_s1`; an incompatible existing run is
kept and the rerun is written to a new timestamped subdirectory:

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

The one-step launcher exposes both `pick_cup` and `stack_cup`. It uses an
unconditioned diffusion actor and one-step `rise_temporal_v2` Q/V networks;
stored `actor_condition` labels are provenance only and are not actor inputs in
this recipe. Before training, the launcher revalidates the converted rollout
source and every mixed external-link dataset.

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

For StackCup, the fitting file contains the 44 human training demonstrations,
20 `success_train` rollouts, and 10 `failure_train` rollouts (35,626 windows).
The disjoint validation file contains all five human validation episodes, six
`success_valid` rollouts, and four `failure_valid` rollouts (8,190 windows).
Training evaluates the full held-out split after every epoch, measures actor
loss using EMA weights, and writes the lowest-loss checkpoint to
`best_validation.pt`. The default StackCup recipe uses dynamics weight `0.05`,
actor U-Net and observation-encoder learning rates of `1e-5`, and freezes the
actor, Q, and V observation encoders for the first 1,000 reference-batch
updates. Start it with:

```bash
bash run_rgb_dp_idql.sh stack_cup build_dataset
bash run_rgb_dp_idql.sh stack_cup train_resilient
```

## Stack-cup mixed-data chunk IDQL

The chunk-IDQL launcher exposes the current rollout corpus as `stack_cup`. The
raw source is `/home/ryan/datasets/stack_cup/rollout`. Conversion requires all
40 finalized episodes, the deployed epoch-200 checkpoint identity, the complete
published checksum manifest, and the exact 26-success / 14-failure outcome
partition. The checkpoint configuration stores DDIM-10, while the collection
server identity proves that this corpus used the runtime DDIM-100 override.

Camera reconstruction preserves the causal 5 Hz image / 20 Hz action contract.
The processed handoff repeats source actions on a wall-clock grid during
DDIM-100 inference gaps; these synthetic repeats are not training samples. The
converter keeps one verified row per immutable source action index and selects
the latest causal image pair at the original source timestamp, with a maximum
allowed image age of 0.5 seconds. The production rollout output has 23,468
retained executed-action transitions: 31 pre-causal startup actions are dropped
and 54 missing internal source indices are recorded without interpolation.

The mixed fitting file is
`datasets/real_robot/stack_cup/idql/stack_cup_chunk_idql_44demo_20success_10failure_ddim100_terminal_success_human_success_condition.hdf5`.
It contains 74 episodes and 35,626 transitions: 18,062 from the 44 human
`train` episodes, 11,710 from 20 successful rollout-train episodes, and 5,854
from 10 failed rollout-train episodes. It uses external HDF5 links and virtual
shifted `next_obs`, so the image data are not copied into the small mixed file.
The chunk launcher additionally builds
`datasets/real_robot/stack_cup/idql/stack_cup_chunk_idql_validation_5demo_6success_4failure_ddim100_terminal_success_human_success_condition.hdf5`,
containing the five held-out human episodes and the 6/4 held-out successful /
failed rollouts (8,190 transitions). It is evaluated in full after every chunk
training epoch, with the best held-out actor checkpoint retained as
`best_validation.pt`.

Build or revalidate the rollout and mixed datasets through the chunk launcher:

```bash
bash run_rgb_dp_chunk_idql.sh stack_cup build_dataset
```

If the raw rollout handoff is not mounted, an existing converted rollout file
is accepted only after an output-only audit of its embedded immutable manifest,
checkpoint identity, exact episode/mask counts, normalized actions,
reward/terminal semantics, observation shapes, and per-row timing provenance.
Raw source hashes are additionally rechecked whenever the rollout directory is
available. Set `REAL_ROBOT_ROLLOUT_OUTPUT_ONLY_VALIDATION=1` to request this
mode explicitly; an explicit `REAL_ROBOT_ROLLOUT_SOURCE_ROOT` override remains
fail-closed and is never silently downgraded to output-only validation.

Start the default chunked run (joint actor and critic, not separate training):

```bash
bash run_rgb_dp_chunk_idql.sh stack_cup train_chunk_idql
```

It initializes the actor from
`trained_models/real_robot/stack_cup_rgb_dp/stack_cup_rgb_dp_ddim_s1/20260902111545/models/model_epoch_50.pth`,
uses `rise_temporal_v2`, and uses `robot0_gripper_state` for critic late fusion.
Generic simulation eval and collection stages are rejected for this
real-robot task.

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
