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

To exercise the full model and checkpoint path without starting a production
run:

```bash
/home/ryan/miniconda3/envs/robomimic_stable/bin/python -B \
  scripts/real_robot/run_pick_cup_rgb_dp_baseline.py \
  --stages prepare train --smoke
```

The converted dataset already lives in `datasets/real_robot/pick_cup`. To build
it again from the source package, use `--stages dataset prepare`; existing valid
shards are reused only when their conversion settings match the request. Add
`--force-dataset` only when intentionally replacing both shards.

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
  tests.real_robot.test_pick_cup_rgb_dp_baseline -v
```

The converter also writes `datasets/real_robot/pick_cup/conversion_summary.json`
with episode counts, sample counts, causal prefix drops, and maximum image ages.
`dataset_commit.json` is the publication marker; the launcher rejects missing or
mixed shard generations. Training configs also fingerprint each shard, so an old
checkpoint is never silently reused after data or hyperparameters change.
