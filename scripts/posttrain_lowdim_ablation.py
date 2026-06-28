#!/usr/bin/env python3
"""Prepare, train, and evaluate low-dim post-training ablations for robomimic Lift.

This script is intentionally small and explicit. It uses an imperfect BC checkpoint
(epoch 50) as the deployment policy and compares continued post-training on:
  1) original human demos only
  2) collected successful rollouts only
  3) original demos + collected successful rollouts
  4) original demos + collected success + a small amount of failed rollout data

The rollout HDF5 is not copied. Instead, success/failure masks are stored under
`mask/success` and `mask/failure`, and configs refer to those filters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/ryan/miniconda3/envs/robomimic_clean/bin/python')

ORIGINAL_DEMOS = ROOT / 'datasets/lift/ph/low_dim_v15.hdf5'
ROLLOUT_LOWDIM = ROOT / 'rollouts/lift_bc_epoch50_rollouts_500_lowdim.hdf5'
FILTERED_FAILURE = ROOT / 'rollouts/lift_bc_epoch50_failure_segments_stage_filtered.hdf5'
INIT_CKPT = ROOT / 'trained_models/lift_ph_lowdim_bc_full_20260626_session/20260626214440/models/model_epoch_50.pth'
CONFIG_DIR = ROOT / 'robomimic/exps/templates/posttrain_ablation_lowdim'
OUTPUT_DIR = ROOT / 'trained_models/posttrain_ablation_lowdim_epoch50'
RESULTS_DIR = ROOT / 'rollouts/posttrain_ablation_eval'

COMMON_ENV = {
    'MPLCONFIGDIR': '/tmp/matplotlib',
    'MUJOCO_GL': 'egl',
    'PYOPENGL_PLATFORM': 'egl',
    'NUMBA_DISABLE_JIT': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'TORCH_COMPILE_DISABLE': '1',
    'TORCHDYNAMO_DISABLE': '1',
}

OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object',
]

ABLATIONS = {
    'demo_only': {
        'desc': 'continue BC from epoch-50 on original 200 human demos only',
        'data': [
            {'path': str(ORIGINAL_DEMOS), 'weight': 1.0},
        ],
        'normalize_weights_by_ds_size': False,
    },
    'success_only': {
        'desc': 'BC from epoch-50 on collected successful deployment rollouts only',
        'data': [
            {'path': str(ROLLOUT_LOWDIM), 'filter_key': 'success', 'weight': 1.0},
        ],
        'normalize_weights_by_ds_size': False,
    },
    'demo_success': {
        'desc': 'BC from epoch-50 on original demos plus collected successful rollouts, equal source sampling',
        'data': [
            {'path': str(ORIGINAL_DEMOS), 'weight': 1.0},
            {'path': str(ROLLOUT_LOWDIM), 'filter_key': 'success', 'weight': 1.0},
        ],
        'normalize_weights_by_ds_size': True,
    },
    'demo_success_fail025': {
        'desc': 'BC from epoch-50 on demos + successes + lightly weighted failed rollouts',
        'data': [
            {'path': str(ORIGINAL_DEMOS), 'weight': 1.0},
            {'path': str(ROLLOUT_LOWDIM), 'filter_key': 'success', 'weight': 1.0},
            {'path': str(ROLLOUT_LOWDIM), 'filter_key': 'failure', 'weight': 0.25},
        ],
        'normalize_weights_by_ds_size': True,
    },
    'demo_success_filtered_fail025': {
        'desc': 'BC from epoch-50 on demos + successes + privileged stage-filtered failure segments',
        'data': [
            {'path': str(ORIGINAL_DEMOS), 'weight': 1.0},
            {'path': str(ROLLOUT_LOWDIM), 'filter_key': 'success', 'weight': 1.0},
            {'path': str(FILTERED_FAILURE), 'weight': 0.25},
        ],
        'normalize_weights_by_ds_size': True,
    },
}


def env():
    e = os.environ.copy()
    e.update(COMMON_ENV)
    return e


def run(cmd: list[str], log_path: Path | None = None):
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    if log_path is None:
        return subprocess.run(cmd, cwd=ROOT, env=env(), check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def create_filter_key(hdf5_path: Path, demo_keys: list[str], key_name: str):
    with h5py.File(hdf5_path, 'a') as f:
        if 'mask' not in f:
            f.create_group('mask')
        k = f'mask/{key_name}'
        if k in f:
            del f[k]
        f[k] = np.array(demo_keys, dtype='S')
        lengths = [int(f[f'data/{d}'].attrs['num_samples']) for d in demo_keys]
    return lengths


def prepare_masks():
    if not ROLLOUT_LOWDIM.exists():
        raise FileNotFoundError(ROLLOUT_LOWDIM)
    success, failure, all_demos = [], [], []
    returns, lengths = [], []
    with h5py.File(ROLLOUT_LOWDIM, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
        for d in demos:
            g = f[f'data/{d}']
            ret = float(np.sum(g['rewards'][:]))
            n = int(g.attrs['num_samples'])
            all_demos.append(d)
            returns.append(ret)
            lengths.append(n)
            if ret > 0:
                success.append(d)
            else:
                failure.append(d)
    create_filter_key(ROLLOUT_LOWDIM, all_demos, 'all_rollouts')
    create_filter_key(ROLLOUT_LOWDIM, success, 'success')
    create_filter_key(ROLLOUT_LOWDIM, failure, 'failure')
    summary = {
        'path': str(ROLLOUT_LOWDIM),
        'num_rollouts': len(all_demos),
        'num_success': len(success),
        'num_failure': len(failure),
        'success_rate': len(success) / max(1, len(all_demos)),
        'total_samples': int(np.sum(lengths)),
        'mean_horizon': float(np.mean(lengths)),
        'mean_return': float(np.mean(returns)),
        'success_filter_key': 'success',
        'failure_filter_key': 'failure',
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'dataset_mask_summary.json').write_text(json.dumps(summary, indent=4))
    print(json.dumps(summary, indent=4))
    return summary


def make_config(name: str, total_epochs: int, steps_per_epoch: int, batch_size: int, lr: float, tag: str = 'run1'):
    spec = ABLATIONS[name]
    cfg = {
        'algo_name': 'bc',
        'experiment': {
            'name': f'bc_epoch50_posttrain_{name}_{tag}',
            'validate': False,
            'logging': {
                'terminal_output_to_txt': True,
                'log_tb': True,
                'log_wandb': False,
                'wandb_proj_name': 'debug',
            },
            'save': {
                'enabled': True,
                'every_n_seconds': None,
                'every_n_epochs': 50,
                'epochs': [total_epochs],
                'on_best_validation': False,
                'on_best_rollout_return': False,
                'on_best_rollout_success_rate': False,
            },
            'epoch_every_n_steps': steps_per_epoch,
            'validation_epoch_every_n_steps': 10,
            'env': None,
            'additional_envs': None,
            'render': False,
            'render_video': False,
            'keep_all_videos': False,
            'video_skip': 5,
            'rollout': {
                'enabled': False,
                'n': 50,
                'horizon': 400,
                'rate': 50,
                'warmstart': 0,
                'terminate_on_success': True,
            },
            'env_meta_update_dict': {},
            'ckpt_path': str(INIT_CKPT),
        },
        'train': {
            'data': spec['data'],
            'output_dir': str(OUTPUT_DIR),
            'normalize_weights_by_ds_size': spec['normalize_weights_by_ds_size'],
            'num_data_workers': 0,
            # low_dim keeps memory down and is supported by MetaDataset.
            'hdf5_cache_mode': 'low_dim',
            'hdf5_use_swmr': True,
            'hdf5_load_next_obs': False,
            'hdf5_normalize_obs': False,
            'hdf5_filter_key': None,
            'hdf5_validation_filter_key': None,
            'seq_length': 1,
            'pad_seq_length': True,
            'frame_stack': 1,
            'pad_frame_stack': True,
            'dataset_keys': ['actions', 'rewards', 'dones'],
            'action_keys': ['actions'],
            'action_config': {'actions': {'normalization': None}},
            'goal_mode': None,
            'cuda': True,
            'batch_size': batch_size,
            'num_epochs': total_epochs,
            'seed': 11,
            'max_grad_norm': None,
        },
        'algo': {
            'optim_params': {
                'policy': {
                    'optimizer_type': 'adam',
                    'learning_rate': {
                        'initial': lr,
                        'decay_factor': 0.1,
                        'epoch_schedule': [],
                        'scheduler_type': 'multistep',
                    },
                    'regularization': {'L2': 0.0},
                }
            },
            'loss': {'l2_weight': 1.0, 'l1_weight': 0.0, 'cos_weight': 0.0},
            'actor_layer_dims': [1024, 1024],
            'gaussian': {
                'enabled': False,
                'fixed_std': False,
                'init_std': 0.1,
                'min_std': 0.01,
                'std_activation': 'softplus',
                'low_noise_eval': True,
            },
            'gmm': {
                'enabled': False,
                'num_modes': 5,
                'min_std': 0.0001,
                'std_activation': 'softplus',
                'low_noise_eval': True,
            },
            'vae': {
                'enabled': False,
                'latent_dim': 14,
                'latent_clip': None,
                'kl_weight': 1.0,
                'decoder': {'is_conditioned': True, 'reconstruction_sum_across_elements': False},
                'prior': {
                    'learn': False,
                    'is_conditioned': False,
                    'use_gmm': False,
                    'gmm_num_modes': 10,
                    'gmm_learn_weights': False,
                    'use_categorical': False,
                    'categorical_dim': 10,
                    'categorical_gumbel_softmax_hard': False,
                    'categorical_init_temp': 1.0,
                    'categorical_temp_anneal_step': 0.001,
                    'categorical_min_temp': 0.3,
                },
                'encoder_layer_dims': [300, 400],
                'decoder_layer_dims': [300, 400],
                'prior_layer_dims': [300, 400],
            },
            'rnn': {'enabled': False, 'horizon': 10, 'hidden_dim': 400, 'rnn_type': 'LSTM', 'num_layers': 2, 'open_loop': False, 'kwargs': {'bidirectional': False}},
            'transformer': {
                'enabled': False,
                'context_length': 10,
                'embed_dim': 512,
                'num_layers': 6,
                'num_heads': 8,
                'emb_dropout': 0.1,
                'attn_dropout': 0.1,
                'block_output_dropout': 0.1,
                'sinusoidal_embedding': False,
                'activation': 'gelu',
                'supervise_all_steps': False,
                'nn_parameter_for_timesteps': True,
                'pred_future_acs': False,
            },
        },
        'observation': {
            'modalities': {
                'obs': {'low_dim': OBS_KEYS, 'rgb': [], 'depth': [], 'scan': []},
                'goal': {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []},
            },
            'encoder': {
                'low_dim': {'core_class': None, 'core_kwargs': {}, 'obs_randomizer_class': None, 'obs_randomizer_kwargs': {}},
                'rgb': {'core_class': 'VisualCore', 'core_kwargs': {}, 'obs_randomizer_class': None, 'obs_randomizer_kwargs': {}},
                'depth': {'core_class': 'VisualCore', 'core_kwargs': {}, 'obs_randomizer_class': None, 'obs_randomizer_kwargs': {}},
                'scan': {'core_class': 'ScanCore', 'core_kwargs': {}, 'obs_randomizer_class': None, 'obs_randomizer_kwargs': {}},
            },
        },
        'meta': {'hp_base_config_file': None, 'hp_keys': [], 'hp_values': []},
    }
    return cfg


def write_configs(total_epochs: int, steps_per_epoch: int, batch_size: int, lr: float, tag: str = 'run1'):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ABLATIONS:
        cfg = make_config(name, total_epochs, steps_per_epoch, batch_size, lr, tag)
        path = CONFIG_DIR / f'{name}.json'
        path.write_text(json.dumps(cfg, indent=4))
        paths[name] = path
        print(f'wrote {path}')
    return paths


def latest_checkpoint_for(name: str, tag: str = 'run1') -> Path | None:
    exp_name = f'bc_epoch50_posttrain_{name}_{tag}'
    root = OUTPUT_DIR / exp_name
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        return None
    # Prefer last.pth in latest run. Fall back to latest model_epoch_*.pth.
    run_dir = runs[-1]
    last = run_dir / 'models/last.pth'
    if last.exists():
        return last
    models = sorted((run_dir / 'models').glob('model_epoch_*.pth'))
    return models[-1] if models else None


def train_one(name: str, cfg_path: Path):
    log = RESULTS_DIR / f'train_{name}.log'
    cmd = [str(PYTHON), '-m', 'robomimic.scripts.train', '--config', str(cfg_path)]
    run(cmd, log_path=log)


def parse_rollout_stats(text: str):
    marker = 'Average Rollout Stats'
    idx = text.rfind(marker)
    if idx < 0:
        return None
    rest = text[idx + len(marker):]
    m = re.search(r'\{.*?\}', rest, re.S)
    if not m:
        return None
    return json.loads(m.group(0))


def evaluate_one(name: str, ckpt_path: Path, n_rollouts: int, seed: int, horizon: int):
    log = RESULTS_DIR / f'eval_{name}.log'
    cmd = [str(PYTHON), '-m', 'robomimic.scripts.run_trained_agent', '--agent', str(ckpt_path), '--n_rollouts', str(n_rollouts), '--horizon', str(horizon), '--seed', str(seed)]
    print('\n$ ' + ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    log.write_text(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    stats = parse_rollout_stats(proc.stdout)
    if stats is None:
        raise RuntimeError(f'could not parse rollout stats for {name}')
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--names', nargs='+', default=list(ABLATIONS.keys()), choices=list(ABLATIONS.keys()))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-rollouts', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=123)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--eval-initial', action='store_true')
    parser.add_argument('--tag', type=str, default='run1', help='suffix added to experiment names to avoid overwrite prompts')
    args = parser.parse_args()

    if not (args.prepare or args.train or args.eval or args.eval_initial):
        args.prepare = args.train = args.eval = True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        prepare_masks()
        write_configs(args.epochs, args.steps_per_epoch, args.batch_size, args.lr, args.tag)

    cfg_paths = {name: CONFIG_DIR / f'{name}.json' for name in ABLATIONS}

    if args.train:
        for name in args.names:
            train_one(name, cfg_paths[name])

    summary = {}
    if args.eval_initial:
        summary['initial_epoch50'] = evaluate_one('initial_epoch50', INIT_CKPT, args.n_rollouts, args.eval_seed, args.horizon)
    if args.eval:
        for name in args.names:
            ckpt = latest_checkpoint_for(name, args.tag)
            if ckpt is None:
                print(f'[skip] no checkpoint found for {name}', file=sys.stderr)
                continue
            summary[name] = evaluate_one(name, ckpt, args.n_rollouts, args.eval_seed, args.horizon)
            summary[name]['checkpoint'] = str(ckpt)

    if summary:
        out = RESULTS_DIR / 'posttrain_ablation_summary.json'
        out.write_text(json.dumps(summary, indent=4))
        print('\nSummary')
        print(json.dumps(summary, indent=4))
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
