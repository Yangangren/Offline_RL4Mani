"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # extract 84x84 image and depth observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import atexit
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


_CONVERSION_MANIFEST_ATTR = "_robomimic_conversion_manifest"
_CONVERSION_STATUS_ATTR = "_robomimic_conversion_status"
_EPISODE_COMPLETE_ATTR = "_robomimic_conversion_complete"


def _metadata_identity(value):
    """Return a stable, exact identity for XML and episode metadata values."""
    if isinstance(value, bytes):
        return ("bytes", value)
    if value is None:
        return ("none", None)
    return ("str", str(value))


def _close_environment(env):
    """Explicitly release renderer and simulator resources when supported."""
    if env is None:
        return
    try:
        base_env = env.base_env
        close_fn = getattr(base_env, "close", None)
        if callable(close_fn):
            close_fn()
            return
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as exc:
        # Cleanup should be attempted on every exit, but should not hide the
        # conversion result or the original exception.
        print("WARNING: environment cleanup failed: {}".format(exc))


def _conversion_manifest(args):
    source_stat = os.stat(args.dataset)
    return json.dumps(
        dict(
            version=1,
            source_path=os.path.realpath(args.dataset),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            n=args.n,
            shaped=args.shaped,
            done_mode=args.done_mode,
            copy_rewards=args.copy_rewards,
            copy_dones=args.copy_dones,
            camera_names=list(args.camera_names),
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            depth=args.depth,
            exclude_next_obs=args.exclude_next_obs,
            compress=args.compress,
            reuse_identical_models=args.reuse_identical_models,
        ),
        sort_keys=True,
    )


def _close_hdf5(file_handle):
    if file_handle is not None and file_handle.id.valid:
        file_handle.close()


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    actions_abs,
    done_mode,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    obs = env.reset_to(initial_state)

    # maybe add in intrinsics and extrinsics for all cameras
    camera_info = None
    is_robosuite_env = EnvUtils.is_robosuite_env(env=env)
    if is_robosuite_env:
        camera_info = get_camera_info(
            env=env,
            camera_names=camera_names, 
            camera_height=camera_height, 
            camera_width=camera_width,
        )

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    if actions_abs is not None:
        traj["actions_abs"] = np.array(actions_abs)
    
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, camera_info


def get_camera_info(
    env,
    camera_names=None, 
    camera_height=84, 
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    # check for v1.5+ robosuite
    import robosuite
    is_v15 = (robosuite.__version__.split(".")[0] == "1") and (robosuite.__version__.split(".")[1] >= "5")

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(camera_name=cam_name, camera_height=camera_height, camera_width=camera_width)
        R = env.get_camera_extrinsic_matrix(camera_name=cam_name) # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0") or cam_name.startswith("robot1")
            robot_ind = int(cam_name[5])
            if is_v15:
                eef_site_name = env.base_env.robots[robot_ind].composite_controller.part_controllers["right"].ref_name
            else:
                eef_site_name = env.base_env.robots[robot_ind].controller.eef_name
            eef_pos = np.array(env.base_env.sim.data.site_xpos[env.base_env.sim.model.site_name2id(eef_site_name)])
            eef_rot = np.array(env.base_env.sim.data.site_xmat[env.base_env.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
            eef_pose = np.zeros((4, 4)) # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv) # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # Resolve and validate output paths before allocating MuJoCo / EGL.
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(args.camera_width)
    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    partial_path = output_path + ".partial"
    if os.path.realpath(output_path) == os.path.realpath(args.dataset):
        raise ValueError("output dataset must differ from the input dataset")
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(
            "output dataset already exists at {}. Refusing to overwrite it; "
            "pass --overwrite explicitly if replacement is intended.".format(output_path)
        )
    if args.restart and os.path.exists(partial_path):
        os.remove(partial_path)

    partial_exists = os.path.exists(partial_path)
    if partial_exists and not args.resume:
        raise FileExistsError(
            "partial conversion already exists at {}. Re-run with --resume "
            "or --restart.".format(partial_path)
        )

    manifest = _conversion_manifest(args)

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
        use_depth_obs=args.depth,
    )
    atexit.register(_close_environment, env)

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    atexit.register(_close_hdf5, f)
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # Write to a partial file, and atomically publish only a complete dataset.
    f_out = h5py.File(partial_path, "a" if partial_exists else "w")
    atexit.register(_close_hdf5, f_out)
    if partial_exists:
        stored_manifest = f_out.attrs.get(_CONVERSION_MANIFEST_ATTR, None)
        if isinstance(stored_manifest, bytes):
            stored_manifest = stored_manifest.decode("utf-8")
        if stored_manifest != manifest:
            raise RuntimeError(
                "partial conversion options or source dataset do not match the "
                "current request. Re-run with --restart after checking the paths."
            )
        print("resuming partial output: {}".format(partial_path))
        data_grp = f_out["data"]
    else:
        f_out.attrs[_CONVERSION_MANIFEST_ATTR] = manifest
        f_out.attrs[_CONVERSION_STATUS_ATTR] = "in_progress"
        data_grp = f_out.create_group("data")

    # Drop only a trajectory that was interrupted before its final flush.
    completed_demos = set()
    for ep in list(data_grp.keys()):
        if bool(data_grp[ep].attrs.get(_EPISODE_COMPLETE_ATTR, False)):
            completed_demos.add(ep)
        else:
            del data_grp[ep]
    unexpected_demos = completed_demos.difference(demos)
    if unexpected_demos:
        raise RuntimeError(
            "partial output contains demos outside this request: {}".format(
                sorted(unexpected_demos)
            )
        )

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))
    print("partial file: {}".format(partial_path))
    if completed_demos:
        print(
            "reusing {} completed trajectories from the partial output".format(
                len(completed_demos)
            )
        )

    total_samples = sum(
        int(data_grp[ep].attrs["num_samples"])
        for ep in completed_demos
    )
    loaded_model_key = None
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]
        if ep in completed_demos:
            continue

        source_ep_grp = f["data/{}".format(ep)]

        # prepare initial state to reload from
        states = source_ep_grp["states"][()]
        initial_state = dict(states=states[0])
        model_file = None
        if is_robosuite_env:
            model_file = source_ep_grp.attrs["model_file"]
            ep_meta = source_ep_grp.attrs.get("ep_meta", None)
            model_key = (
                _metadata_identity(model_file),
                _metadata_identity(ep_meta),
            )
            if (
                not args.reuse_identical_models
                or model_key != loaded_model_key
            ):
                initial_state["model"] = model_file
                initial_state["ep_meta"] = ep_meta
                loaded_model_key = model_key

        # extract obs, rewards, dones
        actions = source_ep_grp["actions"][()]
        if "actions_abs" in source_ep_grp:
            actions_abs = source_ep_grp["actions_abs"][()]
        else:
            actions_abs = None
        traj, camera_info = extract_trajectory(
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            actions_abs=actions_abs,
            done_mode=args.done_mode,
            camera_names=args.camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = source_ep_grp["rewards"][()]
        if args.copy_dones:
            traj["dones"] = source_ep_grp["dones"][()]

        # store transitions

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        if "actions_abs" in traj:
            ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
        for k in traj["obs"]:
            if args.compress:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
            else:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            if not args.exclude_next_obs:
                if args.compress:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # copy action dict (if applicable)
        if "action_dict" in source_ep_grp:
            action_dict = source_ep_grp["action_dict"]
            for k in action_dict:
                ep_data_grp.create_dataset("action_dict/{}".format(k), data=np.array(action_dict[k][()]))

        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = model_file # model xml for this episode
        if "ep_meta" in source_ep_grp.attrs:
            ep_data_grp.attrs["ep_meta"] = source_ep_grp.attrs["ep_meta"]
        for attr_name in (
            "episode_return",
            "policy_success",
            "source_shard",
            "source_demo",
            "env_seed",
            "policy_seed",
            "env_index",
        ):
            if attr_name in source_ep_grp.attrs:
                ep_data_grp.attrs[attr_name] = source_ep_grp.attrs[attr_name]
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode

        if camera_info is not None:
            assert is_robosuite_env
            ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

        total_samples += traj["actions"].shape[0]
        ep_data_grp.attrs[_EPISODE_COMPLETE_ATTR] = True
        f_out.flush()

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f_out:
        del f_out["mask"]
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    f_out.attrs[_CONVERSION_STATUS_ATTR] = "complete"
    f_out.flush()

    f.close()
    f_out.close()
    _close_environment(env)
    atexit.unregister(_close_hdf5)
    atexit.unregister(_close_environment)
    os.replace(partial_path, output_path)
    print("Wrote {} trajectories to {}".format(len(demos), output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth", 
        action='store_true',
        help="(optional) use depth observations for each camera",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching partial conversion, if one exists",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard a partial conversion and start it again from the first trajectory",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing completed output dataset",
    )

    parser.add_argument(
        "--reuse-identical-models",
        action="store_true",
        help=(
            "avoid repeated XML reloads for consecutive trajectories with "
            "identical model XML and episode metadata"
        ),
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
