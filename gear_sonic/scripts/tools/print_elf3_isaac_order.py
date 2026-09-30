from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)

simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from gear_sonic.envs.manager_env.robots.elf3 import ELF3_CFG


sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=0.01)
)

robot_cfg = ELF3_CFG.replace(
    prim_path="/World/Robot"
)

robot = Articulation(robot_cfg)

sim.reset()

print("========== IsaacLab DOF ==========")

for i, name in enumerate(robot.joint_names):
    print(i, name)

print("\n========== IsaacLab BODY ==========")

for i, name in enumerate(robot.body_names):
    print(i, name)

simulation_app.close()