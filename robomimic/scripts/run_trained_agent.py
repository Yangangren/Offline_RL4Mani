"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for both environment and policy rollouts

    env_seed (int): if provided, set NumPy / Python RNG seed for environment resets

    policy_seed (int): if provided, set Torch RNG seed for stochastic policy sampling

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import random
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    return_state=False,
    return_traj=True,
    camera_names=None,
):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        return_state (bool): if True, return simulator states for writing a rollout dataset.
        return_traj (bool): if True, collect and return actions, rewards, dones, and
            any requested simulator states or observations. Evaluation-only rollouts
            disable this to avoid retaining unnecessary Torch-backed NumPy arrays.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict or None): rollout trajectory, or None when return_traj is False
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    assert return_traj or not (return_obs or return_state)

    policy.start_episode()
    obs = env.reset()

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[]) if return_traj else None
    state_dict = None
    if return_state:
        state_dict = env.get_state()
        traj.update(states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # Collect transitions only when a caller needs the trajectory. Copy
            # actions so collected datasets never retain Torch-owned NumPy storage.
            if return_traj:
                traj["actions"].append(np.asarray(act).copy())
                traj["rewards"].append(r)
                traj["dones"].append(done)
                if return_state:
                    traj["states"].append(state_dict["states"])
                if return_obs:
                    traj["obs"].append(obs)
                    traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if return_state:
                state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_traj and return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    if return_traj:
        for k in traj:
            if k == "initial_state_dict":
                continue
            if isinstance(traj[k], dict):
                for kp in traj[k]:
                    traj[k][kp] = np.array(traj[k][kp])
            else:
                traj[k] = np.array(traj[k])

    return stats, traj


def configure_soft_reset(env):
    """Use sim.reset() between rollouts without rebuilding MuJoCo or EGL."""
    base_env = env.unwrapped if isinstance(env, EnvWrapper) else env
    backend = getattr(base_env, "env", None)
    if backend is not None and hasattr(backend, "hard_reset"):
        backend.hard_reset = False


def close_environment(env):
    """Close the first environment layer that exposes a cleanup method."""
    base_env = env.unwrapped if isinstance(env, EnvWrapper) else env
    backend = getattr(base_env, "env", None)
    for candidate in (env, base_env, backend):
        close = getattr(candidate, "close", None)
        if callable(close):
            close()
            return


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )
    configure_soft_reset(env)

    # maybe set seeds. The legacy --seed path sets both streams;
    # --env_seed and --policy_seed allow explicit split control.
    env_seed = args.env_seed if args.env_seed is not None else args.seed
    policy_seed = args.policy_seed if args.policy_seed is not None else args.seed
    if env_seed is not None:
        random.seed(env_seed)
        np.random.seed(env_seed)
    if policy_seed is not None:
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)

    video_writer = None
    data_writer = None
    write_dataset = (args.dataset_path is not None)
    try:
        if write_video:
            video_writer = imageio.get_writer(args.video_path, fps=20)

        if write_dataset:
            data_writer = h5py.File(args.dataset_path, "w")
            data_grp = data_writer.create_group("data")
            total_samples = 0

        rollout_stats = []
        for i in range(rollout_num_episodes):
            stats, traj = rollout(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                return_state=write_dataset,
                return_traj=write_dataset,
                camera_names=args.camera_names,
            )
            rollout_stats.append(stats)
            print(
                "Rollout {}/{}: return={:.3f}, horizon={}, success={}".format(
                    i + 1,
                    rollout_num_episodes,
                    float(stats["Return"]),
                    int(stats["Horizon"]),
                    int(bool(stats["Success_Rate"])),
                ),
                flush=True,
            )

            if write_dataset:
                # store transitions
                ep_data_grp = data_grp.create_group("demo_{}".format(i))
                ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                if args.dataset_obs:
                    for k in traj["obs"]:
                        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

                # episode metadata
                if "model" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
                if "ep_meta" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
                if env_seed is not None:
                    ep_data_grp.attrs["env_seed"] = int(env_seed)
                if policy_seed is not None:
                    ep_data_grp.attrs["policy_seed"] = int(policy_seed)
                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
                total_samples += traj["actions"].shape[0]

        rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
        avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
        print("Average Rollout Stats")
        print(json.dumps(avg_rollout_stats, indent=4))

        if write_dataset:
            # global metadata
            data_grp.attrs["total"] = total_samples
            data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
            print("Wrote dataset trajectories to {}".format(args.dataset_path))
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            try:
                if data_writer is not None:
                    data_writer.close()
            finally:
                close_environment(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=None,
        help="(optional) seed for environment reset RNGs; overrides --seed for environment",
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        default=None,
        help="(optional) seed for stochastic policy sampling; overrides --seed for policy",
    )

    args = parser.parse_args()
    run_trained_agent(args)

