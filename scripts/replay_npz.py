"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local path to a .npz motion file.")
parser.add_argument(
    "--num_cycles",
    type=int,
    default=0,
    help="Number of replay cycles before exit. 0 means run forever (default).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.motion_file is not None:
        motion_file = pathlib.Path(args_cli.motion_file).expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Local motion file not found: {motion_file}")
        print(f"[INFO]: Using local motion file: {motion_file}")
        motion_file = str(motion_file)
    else:
        if args_cli.registry_name is None:
            raise ValueError("Either --registry_name or --motion_file must be provided.")

        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"

        try:
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            artifact_dir = pathlib.Path(artifact.download())
        except Exception as exc:
            raise RuntimeError(
                "Failed to download motion artifact from W&B. "
                "Please run `wandb login`, verify --registry_name, or use --motion_file for offline replay."
            ) from exc

        print(f"[INFO]: Loaded artifact: {registry_name}")

        # New artifacts are always uploaded as 'motion.npz'.
        # Older artifacts may use the original filename (e.g. 'walk1_subject2.npz').
        # Fall back to any .npz file in the directory so both formats work.
        motion_file = artifact_dir / "motion.npz"
        if not motion_file.is_file():
            npz_files = sorted(artifact_dir.glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(
                    f"No .npz file found in artifact directory: {artifact_dir}. "
                    "Re-upload this artifact with csv_to_npz.py."
                )
            motion_file = npz_files[0]
            print(f"[INFO]: 'motion.npz' not found, using fallback: {motion_file.name}")
        motion_file = str(motion_file)
    print(f"[INFO]: Motion file: {motion_file}")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    total_steps = int(motion.time_step_total)
    print(f"[INFO]: Motion frames: {total_steps}")
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    completed_cycles = 0

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        if reset_ids.any():
            completed_cycles += 1
            if args_cli.num_cycles > 0 and completed_cycles >= args_cli.num_cycles:
                print(f"[INFO]: Reached num_cycles={args_cli.num_cycles}. Exiting replay.")
                simulation_app.close()
                return
            time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        if not args_cli.headless:
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
