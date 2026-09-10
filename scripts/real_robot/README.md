# Real-robot RGB Diffusion Policy pipelines

This folder contains the offline real-robot pipelines for the PickCup,
StackCup, and MoveSpoon tasks. They convert the recorded packages, validate the results,
prepare standard robomimic Diffusion Policy configs, and launch training. ROS
and robot-control deployment are deliberately out of scope.

## Quick start

Run these commands from the repository root with the robomimic environment:

```bash
# Revalidate the published dataset and regenerate the production config.
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py --stages validate prepare

# Train the production baseline (200 epochs by default).
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py --stages prepare train
```

To exercise the 20 Hz model and checkpoint path without starting a
production run:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py \
  --stages prepare train --smoke
```

The converted 20 Hz dataset is
`datasets/real_robot/pick_cup/pick_cup_rgb.hdf5`. To build it again from
`/home/ryan/datasets_new/pick_cup/human`, use `--stages dataset prepare`;
an existing file is reused only when source-backed validation passes. Add
`--force-dataset` only when intentionally replacing that file.

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

## Canonical real-robot IDQL sources

PickCup, StackCup, and MoveSpoon use the same episode-layout schema under
`/home/ryan/datasets_new/{pick_cup,stack_cup,move_spoon}/{human,rollout}`.
Build the request-aligned chunk sources with:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/build_episode_layout_idql_sources.py --task pick_cup
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/build_episode_layout_idql_sources.py --task stack_cup
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/build_episode_layout_idql_sources.py --task move_spoon
```

Use `build_episode_layout_one_step_sources.py --task TASK` for one-step IDQL.
Both converters verify every published checksum plus the exact deployed
checkpoint and runtime contract. All current rollouts record the checkpoint
DDIM-10 sampler plus the deployed DDIM-100 runtime override. PickCup is tied to
its epoch-50 checkpoint; StackCup and MoveSpoon remain tied to epoch 200.
Human actions are kept on their recorded
20 Hz target grid and mapped to the latest causal paired camera frame; only an
initial action prefix before the first camera frame is dropped. The operator-ended
PickCup corpus contains 11,301 actions in 1,440 digest-verified requests: 1,405
full H8 requests, 13 partial terminal requests, and 22 empty final requests.
StackCup and MoveSpoon keep all 600 actions and expose 75 requests per episode.
Chunk sources expose the exact request inputs;
one-step sources hold each request camera pair inside its proposal and insert
the exact per-action pre-command low-dimensional state. The one-step source
masks each nonterminal request-edge transition across the variable inference
pause and retains the final terminal row (9,923 valid PickCup rows in total;
526 per StackCup and MoveSpoon rollout).

The immutable human split is 45/5 for all three tasks. PickCup rollouts split
24/8 train success/failure and 6/2 validation. StackCup uses 20/10 and 6/4;
MoveSpoon uses 20/11 and 5/4. Thus every rollout belongs to either fitting or
validation. The mixed builders use external links, so their small HDF5 files depend
on the canonical source files and validate their identities before training.

## 20 Hz mixed-data chunk IDQL

The three task wrappers share the same mixed builder. It uses
`terminal_success`: a successful episode has its sole reward of 1 on
the final recorded transition, while a failure has zero reward throughout;
both terminate at the recorded episode end. The default chunk-actor condition
is `human_success`: human and successful-rollout rows have condition 1, while
failed-rollout rows have condition 0. Each build or validation prints these
semantics and the positive/negative episode and transition counts. The first
training run uses `pretrained_dp_joint`, initializing
from the task's deployed Diffusion Policy and optimizing the actor jointly
with Q/V; the actor is not frozen.

From the repository root, build the converted rollout and mixed fitting data:

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup build_dataset
```

Running the same stage again without `OVERWRITE_DATASET=1` is the launcher
validation path: it checks raw-source provenance, deterministic masks, source
identities, schema, reward terminals, action/observation shapes, and the mixed
dataset contract without replacing either output.

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

The one-step launcher trains `pick_cup`, `stack_cup`, and `move_spoon`. It uses an
unconditioned diffusion actor and one-step `rise_temporal_v2` Q/V networks;
stored `actor_condition` labels are provenance only and are not actor inputs in
this recipe. Before training, the launcher revalidates the converted rollout
source and every mixed external-link dataset.

Validate or rebuild the data contract, then start the default 50-epoch run:

```bash
bash run_rgb_dp_idql.sh pick_cup build_dataset
bash run_rgb_dp_idql.sh pick_cup train_resilient
```

The PickCup run initializes its trainable diffusion actor from the deployed epoch-50
checkpoint and uses `robot0_gripper_state` as the critic's late-fusion key.
Outputs go to
`trained_models/real_robot/pick_cup_rgb_dp/idql/45demo_24success_8failure_terminal_success_rise_temporal_v2_episode_layout_v1`.
The launcher's generic `eval`, `eval_grid_resilient`, and composed evaluation
stages are rejected for `pick_cup` because they instantiate robomimic
simulation rather than the guarded real-robot client.

For StackCup or MoveSpoon, use the same build and training stages:

```bash
bash run_rgb_dp_idql.sh stack_cup build_dataset
bash run_rgb_dp_idql.sh stack_cup train_resilient
bash run_rgb_dp_idql.sh move_spoon build_dataset
bash run_rgb_dp_idql.sh move_spoon train_resilient
```

The StackCup rollout has an exact pre-command pose and logical gripper state at
every 20 Hz action, but only one exact two-frame camera request per H8 proposal.
The one-step converter therefore holds that exact camera pair across the eight
actions and uses consecutive pre-command low-dimensional states after substep
zero. It excludes every substep-7 bootstrap transition because it crosses the
variable DDIM inference pause, while retaining the final terminal row. This
admits 526 of 600 transitions per rollout. These substep-1-through-7 inputs are
explicitly marked as composite training states, not new request captures.

## Request-aligned mixed-data chunk IDQL

StackCup remains the reference implementation; PickCup and MoveSpoon use the
same conversion, sparse-loader, model, validation, and training contract.
Every task has 40 finalized policy rollouts and 50 human demonstrations.
Conversion verifies the complete checksum manifests, each task's exact
deployed checkpoint, and the runtime contract.

Each StackCup and MoveSpoon policy rollout has 600 recorded normalized actions
grouped into 75 exact H8 proposals; PickCup uses variable operator-ended lengths.
Each proposal stores the original `chunk_XXXX_input.npz` as an
exact two-frame `request_obs` tensor for both cameras and all three low-dimensional
state keys. The sparse loader admits only the 75 proposal starts and directly
uses `request_obs`; it never reconstructs those observations from adjacent
action rows. The next critic observation is the exact next request. Dense
dynamics therefore has one honest target at offset 8; the final proposal target
is marked unavailable. Human demonstrations retain causal action-time
observations. The mixed chunk-IDQL builder admits every selected human action
row as a stride-one H8 start; terminal-adjacent rows are shortened by the
existing action mask. Rollouts remain restricted to their exact recorded
proposal starts. PickCup partial terminal requests are retained as shorter
semi-MDP transitions; the trainer verifies their recorded action count against
the terminal-derived action mask. Empty requests produce no training row.

The diffusion actor still uses its normal full 16-step denoising objective at
each admitted row. There is no `actor_action_loss_mask` and no change to the
Diffusion Policy loss implementation.

The StackCup fitting set has 36,841 stored rows and 21,091 admitted H8 starts:
18,841 stride-one human starts plus 2,250 exact rollout proposals. Its held-out
set has 8,325 stored rows and 3,075 admitted starts. MoveSpoon has 37,104 / 20,829
for fitting and 7,477 / 2,752 held out. The refreshed PickCup human source has
17,127 fitting rows and 1,612 held-out rows. The PickCup mixed fitting set has
26,129 stored rows and 18,256 admitted chunk starts; its held-out set has 3,911
stored rows and 1,901 admitted starts. Validation is evaluated in full after every epoch
using EMA actor weights, with the lowest held-out actor loss saved as
`best_validation.pt`.

Build or revalidate the rollout and mixed datasets through the chunk launcher:

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup build_dataset
bash run_rgb_dp_chunk_idql.sh stack_cup build_dataset
bash run_rgb_dp_chunk_idql.sh move_spoon build_dataset
```

The converter writes separate request-aligned, one-step, and human HDF5 sources.
Later runs validate existing sources and mixed files fail-closed. Set
`OVERWRITE_ROLLOUT_DATASET=1 OVERWRITE_DATASET=1` only to rebuild them from the
immutable source package.

If the raw rollout handoff is not mounted, an existing converted rollout file
is accepted only after an output-only audit of its embedded immutable manifest,
checkpoint identity, exact episode/mask counts, normalized actions,
reward/terminal semantics, request observations, and timing provenance.
Raw source hashes are additionally rechecked whenever the rollout directory is
available. Set `REAL_ROBOT_ROLLOUT_OUTPUT_ONLY_VALIDATION=1` to request this
mode explicitly; an explicit `REAL_ROBOT_ROLLOUT_SOURCE_ROOT` override remains
fail-closed and is never silently downgraded to output-only validation.

Start the default chunked run (joint actor and critic, not separate training):

```bash
bash run_rgb_dp_chunk_idql.sh pick_cup train_chunk_idql_resilient
bash run_rgb_dp_chunk_idql.sh stack_cup train_chunk_idql_resilient
bash run_rgb_dp_chunk_idql.sh move_spoon train_chunk_idql_resilient
```

The default model and evaluation directory names contain `episode_layout_v1`
and `human_stride1`, so an older checkpoint cannot be mistaken for this
dataset revision.

It initializes the actor from
`trained_models/real_robot/stack_cup_rgb_dp/stack_cup_rgb_dp_ddim_s1/20260902111545/models/model_epoch_200.pth`,
uses `rise_temporal_v2`, and uses `robot0_gripper_state` for critic late fusion.
The StackCup default enables dense dynamics with weight `0.05` at offsets 4
and 8. Policy rollouts have an exact new camera request only at offset 8, so
their offset-4 target is explicitly masked; human demonstrations provide valid
targets at both offsets. This avoids training the dynamics head against a
fabricated zero-change rollout image at offset 4.
Actor and critic train jointly; actor U-Net and observation-encoder learning
rates are `1e-5`, and actor, Q, and V observation encoders freeze for the first
1,000 reference-batch updates.
Generic simulation eval and collection stages are rejected for this
real-robot task.

## Data contract

- One canonical PickCup HDF5 contains the fixed 45/5 episode split and is
  shared by baseline DP, IDQL, and chunk IDQL.
- Observations are paired main and wrist RGB images at 96x128, EEF position,
  EEF quaternion in `xyzw` order, and the logical gripper state before the
  current action.
- Actions are six already-normalized Cartesian motion channels plus a dense
  post-action gripper target (`-1` closed, `+1` open).
- Image selection is causal against the actual camera header timestamps, not
  nominal frame times. The refreshed PickCup package uses an explicit 1.0 s
  ceiling because 23 recorded action rows contain real camera stalls between
  0.5 and 0.914 s; StackCup and MoveSpoon retain the 0.5 s ceiling.
- Raw gripper events, source row indices, timestamps, selected frame indices,
  camera stamps, and image ages are retained under each demo's `provenance`
  group.
- Spatially diverse validation masks are deterministic and disjoint from the
  training masks.

## Implementation map

- `build_episode_layout_idql_sources.py --task pick_cup --source-kind human`:
  checksum-backed source audit, causal conversion, fixed split creation, and
  atomic single-file publication.
- `run_pick_cup_rgb_dp_baseline.py`: config generation, standard-loader
  preflight, source-backed validation, and training launch.

The baseline reuses `robomimic/algo/diffusion_policy.py` and the existing
robomimic dataset loader. Its default horizons are observation/action/prediction
`2/8/16`, it uses both cameras with 84x112 random crops, DDIM with 10 inference
steps, EMA, and the full `[256, 512, 1024]` temporal U-Net.

## Verification

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -m unittest \
  tests.test_real_robot_episode_layout_profiles \
  tests.test_train_utils_validation_scheduler -v
```

The canonical HDF5 embeds its source identity and conversion manifest. The
launcher verifies that manifest against the current episode-layout source and
fingerprints the dataset in the generated training config, so an old checkpoint
is never silently reused after data or hyperparameters change.
