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

"""Port of the openai gym mujoco envs to run with dm_control.

In this process / port, the simulators are not the same exactly.
Possible reasons are how the integrators are run --
  deepmind uses mj_step / mj_step1 / mj_step2 where as
  openai just uses mj_step.
  Openai does a compute subtree which I am not doing here

In addition, I am not 100% confident in how often the 'MjData' is synced,
as a result there could be a 1 frame offset.
Finally, dm_control code appears to do resets slightly differently and use
different mj functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings.functions import mjlib
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from six.moves import xrange


class CustomPhysics(mujoco.Physics):

  def step(self, n_sub_steps=1):
    """Advances physics with up-to-date position and velocity dependent fields.

    The actuation can be updated by calling the `set_control` function first.

    Args:
      n_sub_steps: Optional number of times to advance the physics. Default 1.
    """

    # This does not line up to how openai does things but instead to how
    # deepmind does.
    # There are configurations where environments like half cheetah become
    # unstable.
    # This is a rough proxy for what is supposed to happen but not perfect.
    for _ in xrange(n_sub_steps):
      mjlib.mj_step(self.model.ptr, self.data.ptr)

    if self.model.opt.integrator != enums.mjtIntegrator.mjINT_EULER:
      mjlib.mj_step1(self.model.ptr, self.data.ptr)


class MujocoEnv(gym.Env):
  """Superclass MuJoCo environments modified to use deepmind's mujoco wrapper.
  """

  def __init__(self,
               model_path,
               frame_skip,
               max_episode_steps=None,
               reward_threshold=None):
    if not os.path.exists(model_path):
      raise IOError('File %s does not exist' % model_path)

    self.frame_skip = frame_skip
    self.physics = CustomPhysics.from_xml_path(model_path)
    self.camera = mujoco.MovableCamera(self.physics, height=480, width=640)

    self.viewer = None

    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': int(np.round(1.0 / self.dt))
    }

    self.init_qpos = self.physics.data.qpos.ravel().copy()
    self.init_qvel = self.physics.data.qvel.ravel().copy()
    observation, _, done, _ = self.step(np.zeros(self.physics.model.nu))
    assert not done
    self.obs_dim = observation.size

    bounds = self.physics.model.actuator_ctrlrange.copy()
    low = bounds[:, 0]
    high = bounds[:, 1]
    self.action_space = spaces.Box(low, high, dtype=np.float32)

    high = np.inf * np.ones(self.obs_dim)
    low = -high
    self.observation_space = spaces.Box(low, high, dtype=np.float32)

    self.max_episode_steps = max_episode_steps
    self.reward_threshold = reward_threshold

    self.seed()
    self.camera_setup()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset_model(self):
    """Reset the robot degrees of freedom (qpos and qvel).

       Implement this in each subclass.
    """
    raise NotImplementedError()

  def viewer_setup(self):
    """This method is called when the viewer is initialized and after all reset.

        Optionally implement this method, if you need to tinker with camera
        position
        and so forth.
    """
    pass

  def reset(self):
    mjlib.mj_resetData(self.physics.model.ptr, self.physics.data.ptr)
    ob = self.reset_model()
    return ob

  def set_state(self, qpos, qvel):
    assert qpos.shape == (self.physics.model.nq,) and qvel.shape == (
        self.physics.model.nv,)
    assert self.physics.get_state().size == qpos.size + qvel.size
    state = np.concatenate([qpos, qvel], 0)
    with self.physics.reset_context():
      self.physics.set_state(state)

  @property
  def dt(self):
    return self.physics.model.opt.timestep * self.frame_skip

  def do_simulation(self, ctrl, n_frames):
    self.physics.set_control(ctrl)
    for _ in range(n_frames):
      self.physics.step()

  def render(self, mode='human'):
    if mode == 'rgb_array':
      data = self.camera.render()
      return np.copy(data)  # render reuses the same memory space.
    elif mode == 'human':
      raise NotImplementedError(
          'Currently no interactive renderings are allowed.')

  def get_body_com(self, body_name):
    idx = self.physics.model.name2id(body_name, 1)
    return self.physics.data.subtree_com[idx]

  def get_body_comvel(self, body_name):
    # As of MuJoCo v2.0, updates to `mjData->subtree_linvel` will be skipped
    # unless these quantities are needed by the simulation. We therefore call
    # `mj_subtreeVel` to update them explicitly.
    mjlib.mj_subtreeVel(self.physics.model.ptr, self.physics.data.ptr)
    idx = self.physics.model.name2id(body_name, 1)
    return self.physics.data.subtree_linvel[idx]

  def get_body_xmat(self, body_name):
    raise NotImplementedError()

  def state_vector(self):
    return np.concatenate(
        [self.physics.data.qpos.flat, self.physics.data.qvel.flat])

  def get_state(self):
    return np.array(self.physics.data.qpos.flat), np.array(
        self.physics.data.qvel.flat)

  def camera_setup(self):
    pass  # override this to set up camera
