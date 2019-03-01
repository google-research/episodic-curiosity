# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake gym environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


# This class is shared among multiple tests. Please refrain from adding logic
# that is specific to your test in this class. Instead, you should create a new
# local fake env that is specific to your use-case (possibly inheriting from
# this one).
class FakeGymEnv(gym.Env):
  """Fake gym environment."""
  OBSERVATION_HEIGHT = 120
  OBSERVATION_WIDTH = 160
  OBSERVATION_CHANNELS = 3
  OBSERVATION_SHAPE = (OBSERVATION_HEIGHT, OBSERVATION_WIDTH,
                       OBSERVATION_CHANNELS)
  NUM_ACTIONS = 4
  EPISODE_LENGTH = 100

  def __init__(self):
    self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
    self.observation_space = gym.spaces.Box(
        0, 255, self.OBSERVATION_SHAPE, dtype=np.float32)
    self.episode_step = 0

  def seed(self, seed=None):
    pass

  def _observation(self):
    return np.random.randint(
        low=0, high=255, size=self.OBSERVATION_SHAPE, dtype=np.uint8)

  def step(self, action):
    observation = self._observation()
    reward = 0.0
    done = self.episode_step >= self.EPISODE_LENGTH
    self.episode_step += 1
    info = {'position': np.random.uniform(low=0, high=1000, size=[3])}
    return observation, reward, done, info

  def reset(self):
    self.episode_step = 0
    return self._observation()

  def render(self, mode='human'):
    raise NotImplementedError('Rendering not implemented')
