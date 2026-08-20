# Pick-cup RGB Diffusion Policy baseline

This folder contains the offline real-robot pipeline for the pick-cup task. It
converts the recorded package in `/home/ryan/datasets/pick_cup`, validates the
result, prepares a standard robomimic Diffusion Policy config, and launches
training. ROS and robot-control deployment are deliberately out of scope.

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
