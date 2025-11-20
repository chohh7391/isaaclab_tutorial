# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path

USD_DIR = "/home/home/isaaclab_tutorial/tutorials/scripts/4_deploying_policy/usd"

def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:

    w = quat[0]
    xyz = quat[1:]

    t = np.cross(xyz, vec) * 2

    return vec - w * t + np.cross(xyz, t)

class Go2WFlatTerrainPolicy(PolicyController):
    """The H1 Humanoid running Flat Terrain Policy Locomotion Policy"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "go2w",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize H1 robot and import flat terrain policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """ 
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            usd_path = f"{USD_DIR}/go2w/go2w.usd"
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        self.load_policy(
            "/home/home/isaaclab_tutorial/tutorials/scripts/4_deploying_policy/pretraind_policy/go2w_rough/policy.pt",
            "/home/home/isaaclab_tutorial/tutorials/scripts/4_deploying_policy/pretraind_policy/go2w_rough/env.yaml",
        )
        self._joint_pos_action_scales = {
            "leg_joint": 0.25, "hip_joint": 0.125
        }
        self._joint_vel_action_scale = 5.0
        self._previous_action = np.zeros(16)
        self._policy_counter = 0

        self.hip_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"
        ]
        self.leg_joint_names = [
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", 
            "FL_calf_joint", "FR_calf_joint",  "RL_calf_joint", "RR_calf_joint"
        ]
        
        self.wheel_joint_names = [
            "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"
        ]

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy.

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        """
        +---------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (57,)) |
        +-----------+---------------------------------+-----------+
        |   Index   | Name                            |   Shape   |
        +-----------+---------------------------------+-----------+
        |     0     | base_ang_vel                    |    (3,)   |
        |     1     | projected_gravity               |    (3,)   |
        |     2     | velocity_commands               |    (3,)   |
        |     3     | joint_pos                       |   (16,)   |
        |     4     | joint_vel                       |   (16,)   |
        |     5     | actions                         |   (16,)   |
        +-----------+---------------------------------+-----------+
        """

        self.joint_pos_dof_ids = {
            "leg_joint": [],
            "hip_joint": [],
        }
        self.joint_vel_dof_id = []

        for joint_name in self.leg_joint_names:
            self.joint_pos_dof_ids["leg_joint"].append(
                self.robot.get_dof_index(joint_name)
            )
        for joint_name in self.hip_joint_names:
            self.joint_pos_dof_ids["hip_joint"].append(
                self.robot.get_dof_index(joint_name)
            )
        for joint_name in self.wheel_joint_names:
            self.joint_vel_dof_id.append(
                self.robot.get_dof_index(joint_name)
            )

        """
        hip_joint: [0, 1, 2, 3]
        leg_joint: [4, 5, 6, 7, 8, 9, 10, 11]
        self.joint_vel_dof_id: [12, 13, 14, 15]
        """

        pos_IB, q_IB = self.robot.get_world_pose()
        ang_vel_I = self.robot.get_angular_velocity()

        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        """
        <IsaacSim>
        ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
        'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
        'FL_foot_joint', 'FR_foot_joint', 'RL_foot_joint', 'RR_foot_joint']
        <IsaacLab>
        ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
        'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
        'FL_foot_joint', 'FR_foot_joint', 'RL_foot_joint', 'RR_foot_joint']
        """

        # policy terms
        base_ang_vel = quat_apply_inverse(q_IB, ang_vel_I)
        projected_gravity = quat_apply_inverse(q_IB, np.array([0.0, 0.0, -9.81]))

        velocity_commands = command
        joint_pos = current_joint_pos - self.default_pos
        # joint_pos_rel_without_wheel
        joint_pos[self.joint_vel_dof_id] = 0.0
        joint_vel = current_joint_vel * 0.05
        actions = self._previous_action

        # obs
        obs = np.zeros(57)
        obs[0:3] = base_ang_vel
        obs[3:6] = projected_gravity
        obs[6:9] = velocity_commands
        obs[9:25] = joint_pos
        obs[25:41] = joint_vel
        obs[41:57] = actions
        obs = np.clip(obs, -100.0, 100.0)

        return obs

    def forward(self, dt, command):
        """
        Compute the desired articulation action and apply them to the robot articulation.

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        self.action = np.clip(self.action, -100.0, 100.0)

        default_pos = np.array(self.default_pos)
        joint_pos_actions = {
            "leg_joint": self.action[self.joint_pos_dof_ids["leg_joint"]],
            "hip_joint": self.action[self.joint_pos_dof_ids["hip_joint"]]
        }
        joint_vel_action = self.action[self.joint_vel_dof_id]
        
        joint_position_actions = {
            "leg_joint": ArticulationAction(
                joint_positions=default_pos[self.joint_pos_dof_ids["leg_joint"]] + (joint_pos_actions["leg_joint"] * self._joint_pos_action_scales["leg_joint"]),
                joint_indices=self.joint_pos_dof_ids["leg_joint"]
            ),
            "hip_joint": ArticulationAction(
                joint_positions=default_pos[self.joint_pos_dof_ids["hip_joint"]] + (joint_pos_actions["hip_joint"] * self._joint_pos_action_scales["hip_joint"]),
                joint_indices=self.joint_pos_dof_ids["hip_joint"]
            ),
        }
        joint_velocity_action = ArticulationAction(
            joint_velocities= joint_vel_action * self._joint_vel_action_scale,
            joint_indices=self.joint_vel_dof_id
        )


        self.robot.apply_action(joint_position_actions["hip_joint"])
        self.robot.apply_action(joint_position_actions["leg_joint"])
        self.robot.apply_action(joint_velocity_action)

        # dummy_action = ArticulationAction(
        #     joint_positions=self.default_pos,
        # )
        # self.robot.apply_action(dummy_action)

        self._policy_counter += 1

    def initialize(self):
        """
        Overloads the default initialize function to use default articulation root properties in the USD
        """
        return super().initialize()
