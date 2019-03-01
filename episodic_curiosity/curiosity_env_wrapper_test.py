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

"""Test of curiosity_env_wrapper.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from episodic_curiosity import curiosity_env_wrapper
from episodic_curiosity import episodic_memory
from third_party.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import gym
import numpy as np
import tensorflow as tf


class DummyImageEnv(gym.Env):

  def __init__(self):
    self._num_actions = 4
    self._image_shape = (28, 28, 3)
    self._done_prob = 0.01

    self.action_space = gym.spaces.Discrete(self._num_actions)
    self.observation_space = gym.spaces.Box(
        0, 255, self._image_shape, dtype=np.float32)

  def seed(self, seed=None):
    pass

  def step(self, action):
    observation = np.random.normal(size=self._image_shape)
    reward = 0.0
    done = (np.random.rand() < self._done_prob)
    info = {}
    return observation, reward, done, info

  def reset(self):
    return np.random.normal(size=self._image_shape)

  def render(self, mode='human'):
    raise NotImplementedError('Rendering not implemented')


# TODO(damienv): To be removed once the code in third_party
# is compatible with python 2.
class HackDummyVecEnv(DummyVecEnv):

  def step_wait(self):
    for e in range(self.num_envs):
      action = self.actions[e]
      if isinstance(self.envs[e].action_space, gym.spaces.Discrete):
        action = int(action)

      obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = (
          self.envs[e].step(action))
      if self.buf_dones[e]:
        obs = self.envs[e].reset()
      self._save_obs(e, obs)
    return (np.copy(self._obs_from_buf()),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            list(self.buf_infos))


def embedding_similarity(x1, x2):
  assert x1.shape[0] == x2.shape[0]
  epsilon = 1e-6

  # Inner product between the embeddings in x1
  # and the embeddings in x2.
  s = np.sum(x1 * x2, axis=-1)

  s /= np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + epsilon
  return 0.5 * (s + 1.0)


def linear_embedding(m, x):
  # Flatten all but the batch dimension if needed.
  if len(x.shape) > 2:
    x = np.reshape(x, [x.shape[0], -1])
  return np.matmul(x, m)


class EpisodicEnvWrapperTest(tf.test.TestCase):

  def EnvFactory(self):
    return DummyImageEnv()

  def testResizeObservation(self):
    img_grayscale = np.random.randint(low=0, high=256, size=[64, 48, 1])
    img_grayscale = img_grayscale.astype(np.uint8)
    resized_img = curiosity_env_wrapper.resize_observation(img_grayscale,
                                                           [16, 12, 1])
    self.assertAllEqual([16, 12, 1], resized_img.shape)

    img_color = np.random.randint(low=0, high=256, size=[64, 48, 3])
    img_color = img_color.astype(np.uint8)
    resized_img = curiosity_env_wrapper.resize_observation(img_color,
                                                           [16, 12, 1])
    self.assertAllEqual([16, 12, 1], resized_img.shape)
    resized_img = curiosity_env_wrapper.resize_observation(img_color,
                                                           [16, 12, 3])
    self.assertAllEqual([16, 12, 3], resized_img.shape)

  def testEpisodicEnvWrapperSimple(self):
    num_envs = 10
    vec_env = HackDummyVecEnv([self.EnvFactory] * num_envs)

    embedding_size = 16
    vec_episodic_memory = [episodic_memory.EpisodicMemory(
        capacity=1000,
        observation_shape=[embedding_size],
        observation_compare_fn=embedding_similarity)
                           for _ in range(num_envs)]

    mat = np.random.normal(size=[28 * 28 * 3, embedding_size])
    observation_embedding = lambda x, m=mat: linear_embedding(m, x)

    target_image_shape = [14, 14, 1]
    env_wrapper = curiosity_env_wrapper.CuriosityEnvWrapper(
        vec_env, vec_episodic_memory,
        observation_embedding,
        target_image_shape)

    observations = env_wrapper.reset()
    self.assertAllEqual([num_envs] + target_image_shape,
                        observations.shape)

    dummy_actions = [1] * num_envs
    for _ in range(100):
      previous_mem_length = [len(mem) for mem in vec_episodic_memory]
      observations, unused_rewards, dones, unused_infos = (
          env_wrapper.step(dummy_actions))
      current_mem_length = [len(mem) for mem in vec_episodic_memory]

      self.assertAllEqual([num_envs] + target_image_shape,
                          observations.shape)
      for k in range(num_envs):
        if dones[k]:
          self.assertEqual(1, current_mem_length[k])
        else:
          self.assertGreaterEqual(current_mem_length[k],
                                  previous_mem_length[k])


if __name__ == '__main__':
  tf.test.main()
