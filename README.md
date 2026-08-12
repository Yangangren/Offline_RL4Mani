# robomimic

<p align="center">
  <img width="24.0%" src="docs/images/task_lift.gif">
  <img width="24.0%" src="docs/images/task_can.gif">
  <img width="24.0%" src="docs/images/task_tool_hang.gif">
  <img width="24.0%" src="docs/images/task_square.gif">
  <img width="24.0%" src="docs/images/task_lift_real.gif">
  <img width="24.0%" src="docs/images/task_can_real.gif">
  <img width="24.0%" src="docs/images/task_tool_hang_real.gif">
  <img width="24.0%" src="docs/images/task_transport.gif">
 </p>

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates
- [06/20/2025] **v0.5.0**: Diffusion Policy, multi-dataset training, language-conditioned policies, and more! 
- [03/11/2025] **v0.4.0**: support for [robosuite v1.5](https://github.com/ARISE-Initiative/robosuite/tree/v1.5.1) and migrate robomimic datasets to HuggingFace
- [10/11/2023] **v0.3.1**: support for extracting, training on, and visualizing depth observations for robosuite datasets
- [07/03/2023] **v0.3.0**: BC-Transformer and IQL :brain:, support for DeepMind MuJoCo bindings :robot:, pre-trained image reps :eye:, wandb logging :chart_with_upwards_trend:, and more
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder: [[release notes]](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0) [[documentation]](https://robomimic.github.io/docs/v0.2/introduction/overview.html)
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

## Colab quickstart
Get started with a quick colab notebook demo of robomimic without installing anything locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing)


-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<p align="center">
  <img width="50.0%" src="docs/images/core_features.png">
 </p>

<!-- **Standardized Datasets**
- Simulated and real-world tasks
- Multiple environments and robots
- Diverse human-collected and machine-generated datasets

**Suite of Learning Algorithms**
- Imitation Learning algorithms (BC, BC-RNN, HBC)
- Offline RL algorithms (BCQ, CQL, IRIS, TD3-BC)

**Modular Design**
- Low-dim + Visuomotor policies
- Diverse network architectures
- Support for external datasets

**Flexible Workflow**
- Hyperparameter sweep tools
- Dataset visualization tools
- Generating new datasets -->


## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

### Multi-GPU Tool Hang chunk-IDQL

The project-specific chunk-IDQL launcher supports single-node distributed
training through `torchrun`. For an eight-GPU Tool Hang run:

```bash
conda activate robomimic_stable
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ROBOMIMIC_PYTHON="$CONDA_PREFIX/bin/python" \
CHUNK_NUM_GPUS=8 \
CHUNK_BATCH_SIZE=100 \
CHUNK_NUM_WORKERS=6 \
CHUNK_SCHEDULE_REFERENCE_BATCH_SIZE=100 \
bash run_rgb_dp_chunk_idql.sh tool_hang train_chunk_idql_resilient
```

`CHUNK_BATCH_SIZE` and `CHUNK_NUM_WORKERS` are per GPU and are not globally
capped. Thus, this example uses an effective batch of 800 and starts 48 data
workers. The warmup, encoder-freeze, dynamics-ramp, hard-sync, target-tau, and
actor-EMA settings are interpreted against
`CHUNK_SCHEDULE_REFERENCE_BATCH_SIZE` (100 by default). For example, a
1,000-step setting resolves to 125 optimizer steps at global batch 800, keeping
the processed-sample timing at 100,000 rows. Learning rates are deliberately
not scaled automatically; use the existing `CHUNK_ACTOR_*_LR`,
`CHUNK_CRITIC_LR`, `CHUNK_ENCODER_LR`, and `CHUNK_VF_LR` controls.

The sparse chunk loader is enabled by default with `HDF5_CACHE_MODE=low_dim`.
It decodes the two actor-history images and one terminal-aware next image
instead of both complete observation sequences. Set
`CHUNK_SPARSE_LOADER=0` for the legacy dense loader.
`CHUNK_GRADIENT_BUCKET_CAP_MB` controls the asynchronous flat all-reduce
bucket size and defaults to 100 MiB.

Ranks receive separate shuffled sampler shards (with standard padding when the
sample count is not divisible by eight) and synchronized parameter updates;
only rank zero writes TensorBoard data, summaries, and checkpoints. Resume a
new-format distributed checkpoint with the same `CHUNK_NUM_GPUS` value.
Checkpoints made before sample-aware schedules should be passed as
`SOURCE_CHUNK_IDQL_CHECKPOINT` to start a fresh round instead of being
resumed in place.

### Multi-GPU chunk-IDQL evaluation

Resilient evaluation can run independent `(candidate count, seed)` pairs on
separate GPUs. For Tool Hang on eight GPUs:

```bash
conda activate robomimic_stable
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ROBOMIMIC_PYTHON="$CONDA_PREFIX/bin/python" \
EVAL_NUM_GPUS=8 \
bash run_rgb_dp_chunk_idql.sh tool_hang eval_chunk_grid_resilient
```

The parent process dynamically schedules pairs across GPUs and alone writes
the aggregate summary. Each worker binds both PyTorch and MuJoCo EGL to its
assigned physical GPU. `EVAL_NUM_GPUS` defaults to one; use a space-separated
list such as `EVAL_GPU_IDS="2 4 6 7"` to select specific physical devices (all
listed devices are used when `EVAL_NUM_GPUS` is omitted). The same controls
work with `eval_composed_chunk_grid_resilient`. Completed pair and chunk files
retain the existing resume behavior.

## Docker

You can use the `Dockerfile` to easily build a containerized environment for setting up robomimic with Python 3.9, Miniconda, robosuite, and PyTorch (CPU/GPU support).

To build, run:
`docker build -t robomimic .`

To run without GPU (CPU only), run:
`docker run -it robomimic`

To run with GPU (if available), run:
`docker run --gpus all -it robomimic`

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```
