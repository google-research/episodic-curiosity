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

"""Evaluation of a policy on a GYM environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


class DummyVideoWriter(object):

  def add(self, obs):
    pass

  def close(self, filename):
    pass


class PolicyEvaluator(object):
  """Evaluate a policy on a GYM environment."""

  def __init__(self, vec_env,
               metric_callback=None,
               video_filename=None, grayscale=False,
               eval_frequency=25):
    """New policy evaluator.

    Args:
      vec_env: baselines.VecEnv correspond to a vector of GYM environments.
      metric_callback: Function that is given the average reward and the time
        step of the evaluation.
      video_filename: Prefix of filenames used to record video.
      grayscale: Whether the observation is grayscale or color.
      eval_frequency: Only performs evaluation once every eval_frequency times.
    """
    self._vec_env = vec_env
    self._metric_callback = metric_callback
    self._video_filename = video_filename
    self._grayscale = grayscale

    self._eval_count = 0
    self._eval_frequency = eval_frequency
    self._discrete_actions = isinstance(self._vec_env.observation_space,
                                        gym.spaces.Discrete)

  def evaluate(self, model_step_fn, global_step):
    """Evaluate the policy as given by its step function.

    Args:
      model_step_fn: Function which given a batch of observations,
        a batch of policy states and a batch of dones flags returns
        a batch of selected actions and updated policy states.
      global_step: The global step of the training process.
    """
    if self._eval_count % self._eval_frequency != 0:
      self._eval_count += 1
      return
    self._eval_count += 1

    video_writer = DummyVideoWriter()
    if self._video_filename:
      video_filename = '{}_{}.mp4'.format(self._video_filename, global_step)
    else:
      video_filename = 'dummy.mp4'

    # Initial state of the policy.
    # TODO(damienv): make the policy state dimension part of the constructor.
    policy_state_dim = 512
    policy_states = np.zeros((self._vec_env.num_envs, policy_state_dim),
                             dtype=np.float32)

    # Reset the environments before starting the evaluation.
    dones = [False] * self._vec_env.num_envs
    sticky_dones = [False] * self._vec_env.num_envs
    obs = self._vec_env.reset()

    # Evaluation loop.
    total_reward = np.zeros((self._vec_env.num_envs,), dtype=np.float32)
    step_iter = 0
    action_distribution = {}
    while not all(sticky_dones):
      actions, _, policy_states, _ = model_step_fn(obs, policy_states, dones)

      # Update the distribution of actions seen along the trajectory.
      if self._discrete_actions:
        for action in actions:
          if action not in action_distribution:
            action_distribution[action] = 0
          action_distribution[action] += 1

      # Update the states of the environment based on the selected actions.
      obs, rewards, dones, infos = self._vec_env.step(actions)
      step_iter += 1
      for k in range(self._vec_env.num_envs,):
        if not sticky_dones[k]:
          total_reward[k] += rewards[k]
      sticky_dones = [sd or d for (sd, d) in zip(sticky_dones, dones)]

      # Optionally record the frames of the 1st environment.
      if not sticky_dones[0]:
        if infos[0].get('frame') is not None:
          frame = infos[0]['frame']
        else:
          frame = obs[0]
        if self._grayscale:
          video_writer.add(frame[:, :, 0])
        else:
          video_writer.add(frame)

    if self._metric_callback:
      self._metric_callback(np.mean(total_reward), global_step)

    print('Average reward: {}, total reward: {}'.format(np.mean(total_reward),
                                                        total_reward))
    if self._discrete_actions:
      print('Action distribution: {}'.format(action_distribution))
    video_writer.close(video_filename)
