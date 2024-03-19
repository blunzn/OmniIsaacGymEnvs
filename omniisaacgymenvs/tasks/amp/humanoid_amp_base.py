# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid
import torch
import omni.kit.commands
from omni.isaac.core.utils.extensions import enable_extension
# from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
enable_extension("omni.importer.mjcf")
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots.robot import Robot
from omniisaacgymenvs.tasks.base.rl_task import RLTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 28


KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]

class HumanoidAMPBase(RLTask):
      
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self._num_observations = self.get_obs_size()
        self._num_actions = self.get_action_size()
        self._robot_spawn_pos = torch.tensor([0, 0, 1.34])

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]

        self._pd_control = self._task_cfg["env"]["pdControl"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.randomize = self._task_cfg["task"]["randomize"]

        self.debug_viz = self._task_cfg["env"]["enableDebugVis"]
        self.camera_follow = self._task_cfg["env"].get("cameraFollow", False)
        
        self._plane_static_friction = self._task_cfg["env"]["plane"]["staticFriction"]
        self._plane_dynamic_friction = self._task_cfg["env"]["plane"]["dynamicFriction"]
        self._plane_restitution = self._task_cfg["env"]["plane"]["restitution"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self._local_root_obs = self._task_cfg["env"]["localRootObs"]
        self._contact_bodies = self._task_cfg["env"]["contactBodies"]
        self._termination_height = self._task_cfg["env"]["terminationHeight"]
        self._enable_early_termination = self._task_cfg["env"]["enableEarlyTermination"]

        # # get gym GPU state tensors
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

       

        # right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        # left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        # self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        # self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        # self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        # self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        # if self.viewer != None:
        #     self._init_camera()

    def set_up_scene(self, scene) -> None:
        self.create_robot()
        RLTask.set_up_scene(self, scene)
        self.create_views(scene)
        return

    def initialize_views(self, scene):
        RLTask.initialize_views(self, scene)
        if scene.object_exists("robot_view"):
            scene.remove_object("robot_view", registry_only=True)
        self.create_views(scene)

    def create_views(self, scene):
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Humanoid/pelvis/torso", name="robot_view", reset_xform_properties=False
        )
        scene.add(self._robots)
        
    def create_robot(self):
        # # import from MJCF
        # # setting up import configuration:
        # status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        # import_config.set_fix_base(False)
        # import_config.set_make_default_prim(False)

        # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets/')
        # asset_file = "mjcf/amp_humanoid.xml"
        # if "asset" in self._task_cfg["env"]:
        #     asset_file = self._task_cfg["env"]["asset"].get("assetFileName", asset_file)
        # # import MJCF
        # prim_path = self.default_zero_env_path + "/Humanoid"

        # omni.kit.commands.execute(
        #     "MJCFCreateAsset",
        #     mjcf_path=asset_root + asset_file,
        #     import_config=import_config,
        #     prim_path=prim_path
        # )

        # robot = Robot(
        #     prim_path=prim_path, name="Humanoid", translation=self._robot_spawn_pos
        # )

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets/')
        asset_file = "amp/humanoid_amp_instanceable.usd"
        
        prim_path = self.default_zero_env_path + "/Humanoid"
        robot = Humanoid(prim_path=prim_path, usd_path=asset_root+asset_file, name="Humanoid")
        self._sim_config.apply_articulation_settings(
            "Humanoid", get_prim_at_path(prim_path), self._sim_config.parse_actor_config("HumanoidAMP")
        )

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def post_reset(self):
        self.num_dof = self._robots.num_dof
        self._key_body_ids =  torch.tensor(
            [self._robots._body_indices[j] for j in KEY_BODY_NAMES], device=self._device, dtype=torch.long
        )

        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(
            self.num_envs
        )
        self.prev_potentials = self.potentials.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        # self._reset_actors(env_ids)
        # self._refresh_sim_tensors()
        # self._compute_observations(env_ids)
        return

#     def set_char_color(self, col):
#         for i in range(self.num_envs):
#             env_ptr = self.envs[i]
#             handle = self.humanoid_handles[i]

#             for j in range(self.num_bodies):
#                 self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
#                                               gymapi.Vec3(col[0], col[1], col[2]))

#         return

#     def _create_envs(self, num_envs, spacing, num_per_row):
#         lower = gymapi.Vec3(-spacing, -spacing, 0.0)
#         upper = gymapi.Vec3(spacing, spacing, spacing)

#         asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
#         asset_file = "mjcf/amp_humanoid.xml"

#         if "asset" in self.cfg["env"]:
#             #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
#             asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

#         asset_options = gymapi.AssetOptions()
#         asset_options.angular_damping = 0.01
#         asset_options.max_angular_velocity = 100.0
#         asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

#         actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
#         motor_efforts = [prop.motor_effort for prop in actuator_props]
        
#         # create force sensors at the feet
#         right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
#         left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
#         sensor_pose = gymapi.Transform()

#         self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
#         self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

#         self.max_motor_effort = max(motor_efforts)
#         self.motor_efforts = to_torch(motor_efforts, device=self.device)

#         self.torso_index = 0
#         self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
#         self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
#         self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

#         start_pose = gymapi.Transform()
#         start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
#         start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

#         self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

#         self.humanoid_handles = []
#         self.envs = []
#         self.dof_limits_lower = []
#         self.dof_limits_upper = []
        
#         for i in range(self.num_envs):
#             # create env instance
#             env_ptr = self.gym.create_env(
#                 self.sim, lower, upper, num_per_row
#             )
#             contact_filter = 0
            
#             handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

#             self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

#             for j in range(self.num_bodies):
#                 self.gym.set_rigid_body_color(
#                     env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

#             self.envs.append(env_ptr)
#             self.humanoid_handles.append(handle)

#             if (self._pd_control):
#                 dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
#                 dof_prop["driveMode"] = gymapi.DOF_MODE_POS
#                 self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

#         dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
#         for j in range(self.num_dof):
#             if dof_prop['lower'][j] > dof_prop['upper'][j]:
#                 self.dof_limits_lower.append(dof_prop['upper'][j])
#                 self.dof_limits_upper.append(dof_prop['lower'][j])
#             else:
#                 self.dof_limits_lower.append(dof_prop['lower'][j])
#                 self.dof_limits_upper.append(dof_prop['upper'][j])

#         self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
#         self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

#         self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
#         self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
#         if (self._pd_control):
#             self._build_pd_action_offset_scale()

#         return

#     def _build_pd_action_offset_scale(self):
#         num_joints = len(DOF_OFFSETS) - 1
        
#         lim_low = self.dof_limits_lower.cpu().numpy()
#         lim_high = self.dof_limits_upper.cpu().numpy()

#         for j in range(num_joints):
#             dof_offset = DOF_OFFSETS[j]
#             dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

#             if (dof_size == 3):
#                 lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
#                 lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

#             elif (dof_size == 1):
#                 curr_low = lim_low[dof_offset]
#                 curr_high = lim_high[dof_offset]
#                 curr_mid = 0.5 * (curr_high + curr_low)
                
#                 # extend the action range to be a bit beyond the joint limits so that the motors
#                 # don't lose their strength as they approach the joint limits
#                 curr_scale = 0.7 * (curr_high - curr_low)
#                 curr_low = curr_mid - curr_scale
#                 curr_high = curr_mid + curr_scale

#                 lim_low[dof_offset] = curr_low
#                 lim_high[dof_offset] =  curr_high

#         self._pd_action_offset = 0.5 * (lim_high + lim_low)
#         self._pd_action_scale = 0.5 * (lim_high - lim_low)
#         self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
#         self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

#         return

#     def _compute_reward(self, actions):
#         self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
#         return

#     def _compute_reset(self):
#         self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
#                                                    self._contact_forces, self._contact_body_ids,
#                                                    self._rigid_body_pos, self.max_episode_length,
#                                                    self._enable_early_termination, self._termination_height)
#         return

#     def _refresh_sim_tensors(self):
#         self.gym.refresh_dof_state_tensor(self.sim)
#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self.gym.refresh_rigid_body_state_tensor(self.sim)

#         self.gym.refresh_force_sensor_tensor(self.sim)
#         self.gym.refresh_dof_force_tensor(self.sim)
#         self.gym.refresh_net_contact_force_tensor(self.sim)
#         return

#     def _compute_observations(self, env_ids=None):
#         obs = self._compute_humanoid_obs(env_ids)

#         if (env_ids is None):
#             self.obs_buf[:] = obs
#         else:
#             self.obs_buf[env_ids] = obs

#         return

#     def _compute_humanoid_obs(self, env_ids=None):
#         if (env_ids is None):
#             root_states = self._root_states
#             dof_pos = self._dof_pos
#             dof_vel = self._dof_vel
#             key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
#         else:
#             root_states = self._root_states[env_ids]
#             dof_pos = self._dof_pos[env_ids]
#             dof_vel = self._dof_vel[env_ids]
#             key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        
#         obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
#                                             key_body_pos, self._local_root_obs)
#         return obs

#     def _reset_actors(self, env_ids):
#         self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
#         self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

#         env_ids_int32 = env_ids.to(dtype=torch.int32)
#         self.gym.set_actor_root_state_tensor_indexed(self.sim,
#                                                      gymtorch.unwrap_tensor(self._initial_root_states),
#                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

#         self.gym.set_dof_state_tensor_indexed(self.sim,
#                                               gymtorch.unwrap_tensor(self._dof_state),
#                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

#         self.progress_buf[env_ids] = 0
#         self.reset_buf[env_ids] = 0
#         self._terminate_buf[env_ids] = 0
#         return

#     def pre_physics_step(self, actions):
#         self.actions = actions.to(self.device).clone()

#         if (self._pd_control):
#             pd_tar = self._action_to_pd_targets(self.actions)
#             pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
#             self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
#         else:
#             forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
#             force_tensor = gymtorch.unwrap_tensor(forces)
#             self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

#         return

#     def post_physics_step(self):
#         self.progress_buf += 1

#         self._refresh_sim_tensors()
#         self._compute_observations()
#         self._compute_reward(self.actions)
#         self._compute_reset()
        
#         self.extras["terminate"] = self._terminate_buf

#         # debug viz
#         if self.viewer and self.debug_viz:
#             self._update_debug_viz()

#         return

#     def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
#         body_ids = []
#         for body_name in self._contact_bodies:
#             body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
#             assert(body_id != -1)
#             body_ids.append(body_id)

#         body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
#         return body_ids

#     def _action_to_pd_targets(self, action):
#         pd_tar = self._pd_action_offset + self._pd_action_scale * action
#         return pd_tar

#     def _init_camera(self):
#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()
        
#         cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
#                               self._cam_prev_char_pos[1] - 3.0, 
#                               1.0)
#         cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
#                                  self._cam_prev_char_pos[1],
#                                  1.0)
#         self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
#         return

#     def _update_camera(self):
#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
#         cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
#         cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
#         cam_delta = cam_pos - self._cam_prev_char_pos

#         new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
#         new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
#                                   char_root_pos[1] + cam_delta[1], 
#                                   cam_pos[2])

#         self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

#         self._cam_prev_char_pos[:] = char_root_pos
#         return

#     def _update_debug_viz(self):
#         self.gym.clear_lines(self.viewer)
#         return

# #####################################################################
# ###=========================jit functions=========================###
# #####################################################################

# @torch.jit.script
# def dof_to_obs(pose):
#     # type: (Tensor) -> Tensor
#     #dof_obs_size = 64
#     #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
#     dof_obs_size = 52
#     dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
#     num_joints = len(dof_offsets) - 1

#     dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
#     dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
#     dof_obs_offset = 0

#     for j in range(num_joints):
#         dof_offset = dof_offsets[j]
#         dof_size = dof_offsets[j + 1] - dof_offsets[j]
#         joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

#         # assume this is a spherical joint
#         if (dof_size == 3):
#             joint_pose_q = exp_map_to_quat(joint_pose)
#             joint_dof_obs = quat_to_tan_norm(joint_pose_q)
#             dof_obs_size = 6
#         else:
#             joint_dof_obs = joint_pose
#             dof_obs_size = 1

#         dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
#         dof_obs_offset += dof_obs_size

#     return dof_obs

# @torch.jit.script
# def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
#     # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
#     root_pos = root_states[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     root_vel = root_states[:, 7:10]
#     root_ang_vel = root_states[:, 10:13]

#     root_h = root_pos[:, 2:3]
#     heading_rot = calc_heading_quat_inv(root_rot)

#     if (local_root_obs):
#         root_rot_obs = quat_mul(heading_rot, root_rot)
#     else:
#         root_rot_obs = root_rot
#     root_rot_obs = quat_to_tan_norm(root_rot_obs)

#     local_root_vel = my_quat_rotate(heading_rot, root_vel)
#     local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

#     root_pos_expand = root_pos.unsqueeze(-2)
#     local_key_body_pos = key_body_pos - root_pos_expand
    
#     heading_rot_expand = heading_rot.unsqueeze(-2)
#     heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
#     flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
#     flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
#                                                heading_rot_expand.shape[2])
#     local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
#     flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

#     dof_obs = dof_to_obs(dof_pos)

#     obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
#     return obs

# @torch.jit.script
# def compute_humanoid_reward(obs_buf):
#     # type: (Tensor) -> Tensor
#     reward = torch.ones_like(obs_buf[:, 0])
#     return reward

# @torch.jit.script
# def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
#                            max_episode_length, enable_early_termination, termination_height):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
#     terminated = torch.zeros_like(reset_buf)

#     if (enable_early_termination):
#         masked_contact_buf = contact_buf.clone()
#         masked_contact_buf[:, contact_body_ids, :] = 0
#         fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
#         fall_contact = torch.any(fall_contact, dim=-1)

#         body_height = rigid_body_pos[..., 2]
#         fall_height = body_height < termination_height
#         fall_height[:, contact_body_ids] = False
#         fall_height = torch.any(fall_height, dim=-1)

#         has_fallen = torch.logical_and(fall_contact, fall_height)

#         # first timestep can sometimes still have nonzero contact forces
#         # so only check after first couple of steps
#         has_fallen *= (progress_buf > 1)
#         terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

#     return reset, terminated