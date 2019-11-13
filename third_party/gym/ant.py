# coding=utf-8
# The MIT License
#
# Copyright (c) 2016 OpenAI (https://openai.com)
# Copyright (c) 2018 The TF-Agents Authors.
# Copyright (c) 2018 Google LLC (http://google.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Ant environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco.wrapper.mjbindings import enums
from third_party.gym import mujoco_env
from gym import utils
import numpy as np


# pylint: disable=missing-docstring
class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self,
               expose_all_qpos=False,
               expose_body_coms=None,
               expose_body_comvels=None,
               model_path="ant.xml"):
    self._expose_all_qpos = expose_all_qpos
    self._expose_body_coms = expose_body_coms
    self._expose_body_comvels = expose_body_comvels
    self._body_com_indices = {}
    self._body_comvel_indices = {}
    # Settings from
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    mujoco_env.MujocoEnv.__init__(
        self, model_path, 5, max_episode_steps=1000, reward_threshold=6000.0)
    utils.EzPickle.__init__(self)

    self.camera_setup()

  def step(self, a):
    xposbefore = self.get_body_com("torso")[0]
    self.do_simulation(a, self.frame_skip)
    xposafter = self.get_body_com("torso")[0]
    forward_reward = (xposafter - xposbefore) / self.dt
    ctrl_cost = .5 * np.square(a).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(
        np.square(np.clip(self.physics.data.cfrc_ext, -1, 1)))
    survive_reward = 1.0
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    state = self.state_vector()
    notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
    done = not notdone
    ob = self._get_obs()
    return ob, reward, done, dict(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        reward_survive=survive_reward)

  def _get_obs(self):
    if self._expose_all_qpos:
      obs = np.concatenate([
          self.physics.data.qpos.flat,
          self.physics.data.qvel.flat,
          np.clip(self.physics.data.cfrc_ext, -1, 1).flat,
      ])
    else:
      obs = np.concatenate([
          self.physics.data.qpos.flat[2:],
          self.physics.data.qvel.flat,
          np.clip(self.physics.data.cfrc_ext, -1, 1).flat,
      ])

    if self._expose_body_coms is not None:
      for name in self._expose_body_coms:
        com = self.get_body_com(name)
        if name not in self._body_com_indices:
          indices = range(len(obs), len(obs) + len(com))
          self._body_com_indices[name] = indices
        obs = np.concatenate([obs, com])

    if self._expose_body_comvels is not None:
      for name in self._expose_body_comvels:
        comvel = self.get_body_comvel(name)
        if name not in self._body_comvel_indices:
          indices = range(len(obs), len(obs) + len(comvel))
          self._body_comvel_indices[name] = indices
        obs = np.concatenate([obs, comvel])
    return obs

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.physics.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel + self.np_random.randn(self.physics.model.nv) * .1
    self.set_state(qpos, qvel)
    return self._get_obs()

  def camera_setup(self):
    # pylint: disable=protected-access
    self.camera._render_camera.type_ = enums.mjtCamera.mjCAMERA_TRACKING
    # pylint: disable=protected-access
    self.camera._render_camera.trackbodyid = 1
    # pylint: disable=protected-access
    self.camera._render_camera.distance = self.physics.model.stat.extent

  @property
  def body_com_indices(self):
    return self._body_com_indices

  @property
  def body_comvel_indices(self):
    return self._body_comvel_indices
