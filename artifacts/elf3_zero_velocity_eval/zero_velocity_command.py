"""Evaluation-only diagnostic: preserve the reference reset pose, clear actual velocity."""

import torch
from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand, TrackingCommandCfg


class ZeroVelocityTrackingCommand(TrackingCommand):
    def _resample_command(self, env_ids):
        # Retain normal motion selection, cursor, pose and reference construction.
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or not self.is_evaluating:
            return

        # Only change the physical robot's initial velocities. The walk reference
        # and its future velocities remain untouched.
        joint_pos = self.robot.data.joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
        )
        root_velocity = torch.zeros(
            (len(env_ids), 6), device=joint_pos.device, dtype=joint_pos.dtype
        )
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        if not getattr(self, "_zero_velocity_diagnostic_logged", False):
            print("[ZeroVelocityEval] Reference reset pose retained; actual joint/root velocities cleared.")
            self._zero_velocity_diagnostic_logged = True


@configclass
class ZeroVelocityTrackingCommandCfg(TrackingCommandCfg):
    class_type: type = ZeroVelocityTrackingCommand
