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

"""Tests for r_network_training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import absltest
from episodic_curiosity import r_network_training
from episodic_curiosity.constants import Const
from episodic_curiosity.r_network_training import create_training_data_from_episode_buffer_v123
from episodic_curiosity.r_network_training import create_training_data_from_episode_buffer_v4
from episodic_curiosity.r_network_training import generate_negative_example
import numpy as np


_OBSERVATION_SHAPE = (120, 160, 3)


class TestRNetworkTraining(absltest.TestCase):

  def test_generate_negative_example(self):
    max_action_distance = 5
    len_episode_buffer = 100
    for _ in range(1000):
      buffer_position = np.random.randint(low=0, high=len_episode_buffer)
      first, second = generate_negative_example(buffer_position,
                                                len_episode_buffer,
                                                max_action_distance)
      self.assertGreater(
          abs(second - first),
          Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance)
      self.assertGreaterEqual(second, 0)
      self.assertLess(second, len_episode_buffer)

  def test_generate_negative_example2(self):
    max_action_distance = 5
    range_max = 5 * Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance
    for buffer_position in range(0, range_max):
      for buffer_length in range(buffer_position + 1, 3 * range_max):
        # Mainly check that it does not raise an exception.
        new_pos, index = generate_negative_example(buffer_position,
                                                   buffer_length,
                                                   max_action_distance)
        msg = 'buffer_pos={}, buffer_length={}'.format(buffer_position,
                                                       buffer_length)
        self.assertLess(new_pos, buffer_length, msg)
        self.assertGreaterEqual(new_pos, 0, msg)
        if index is not None:
          self.assertLess(index, buffer_length, msg)
          self.assertGreaterEqual(index, 0, msg)

  def test_create_training_data_from_episode_buffer_v1(self):
    # Make the test deterministic.
    random.seed(7)
    max_action_distance_mode = 'v1_affect_num_training_examples'
    max_action_distance = 5
    len_episode_buffer = 1000
    episode_buffer = [(np.zeros(_OBSERVATION_SHAPE), {
        'position': i
    }) for i in range(len_episode_buffer)]
    x1, x2, labels = create_training_data_from_episode_buffer_v123(
        episode_buffer, max_action_distance, max_action_distance_mode)
    self.assertEqual(len(x1), len(x2))
    self.assertEqual(len(labels), len(x1))
    # 4 is the average of 1 + randint(1, 5).
    self.assertGreater(len(x1), len_episode_buffer / 4 * 0.8)
    self.assertLess(len(x1), len_episode_buffer / 4 * 1.2)
    for x in [x1, x2]:
      self.assertTupleEqual(x[0][0].shape, _OBSERVATION_SHAPE)
    max_previous_pos = -1
    for xx1, xx2, label in zip(x1, x2, labels):
      if not label:
        continue
      obs1, info1 = xx1
      del obs1  # unused
      pos1 = info1['position']
      obs2, info2 = xx2
      del obs2  # unused
      pos2 = info2['position']
      assert max_previous_pos < pos1, (
          'In v1 mode, intervals of positive examples should have no overlap.')
      assert max_previous_pos < pos2, (
          'In v1 mode, intervals of positive examples should have no overlap.')
      max_previous_pos = max(pos1, pos2)
    self._check_example_pairs(x1, x2, labels, max_action_distance)

  def test_create_training_data_from_episode_buffer_v2(self):
    # Make the test deterministic.
    random.seed(7)
    max_action_distance_mode = 'v2_fixed_num_training_examples'
    max_action_distance = 2
    len_episode_buffer = 1000
    episode_buffer = [(np.zeros(_OBSERVATION_SHAPE), {
        'position': i
    }) for i in range(len_episode_buffer)]
    x1, x2, labels = create_training_data_from_episode_buffer_v123(
        episode_buffer, max_action_distance, max_action_distance_mode)
    self.assertEqual(len(x1), len(x2))
    self.assertEqual(len(labels), len(x1))
    # 4 is the average of 1 + randint(1, 5), where 5 is used in the v2 sampling
    # algorithm in order to keep the same number of training example regardless
    # of max_action_distance.
    self.assertGreater(len(x1), len_episode_buffer / 4 * 0.8)
    self.assertLess(len(x1), len_episode_buffer / 4 * 1.2)
    self._check_example_pairs(x1, x2, labels, max_action_distance)

  def test_create_training_data_from_episode_buffer_v4(self):
    # Make the test deterministic.
    random.seed(7)
    for avg_num_examples_per_env_step in (0.2, 0.5, 1, 2):
      max_action_distance = 5
      len_episode_buffer = 1000
      episode_buffer = [(np.zeros(_OBSERVATION_SHAPE), {
          'position': i
      }) for i in range(len_episode_buffer)]
      x1, x2, labels = create_training_data_from_episode_buffer_v4(
          episode_buffer, max_action_distance, avg_num_examples_per_env_step)
      self.assertEqual(len(x1), len(x2))
      self.assertEqual(len(labels), len(x1))
      num_positives = len([label for label in labels if label == 1])
      self._assert_within(num_positives, len(x1) // 2, 5)
      self._assert_within(
          len(labels),
          len(episode_buffer) * avg_num_examples_per_env_step, 5)
      self._check_example_pairs(x1, x2, labels, max_action_distance)

  def test_create_training_data_from_episode_buffer_too_short(self):
    max_action_distance = 5
    buff = [np.zeros(_OBSERVATION_SHAPE)] * max_action_distance
    # Repeat the test multiple times. create_training_data_from_episode_buffer
    # uses randomness, so we want to hit the case where it tries to generate
    # negative examples.
    for _ in range(50):
      _, _, labels = create_training_data_from_episode_buffer_v123(
          buff, max_action_distance, mode='v1_affect_num_training_examples')
      for label in labels:
        # Not enough buffer to generate negative examples, so we should get only
        # (but possibly none) positive examples.
        self.assertEqual(label, 1)

  def _check_example_pairs(self, x1, x2, labels, max_action_distance):
    for xx1, xx2, label in zip(x1, x2, labels):
      obs1, info1 = xx1
      del obs1  # unused
      pos1 = info1['position']
      obs2, info2 = xx2
      del obs2  # unused
      pos2 = info2['position']
      diff = abs(pos1 - pos2)
      if label:
        self.assertLessEqual(diff, max_action_distance)
      else:
        self.assertGreater(
            diff, Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance)

  def _assert_within(self, value, expected, percentage):
    self.assertGreater(
        value,
        expected * (100 - percentage) / 100, '{} vs {} within {}%'.format(
            value, expected, percentage))
    self.assertLess(value,
                    expected * (100 + percentage) / 100,
                    '{} vs {} within {}%'.format(value, expected, percentage))

  def fit_generator(self,
                    batch_gen, steps_per_epoch, epochs,
                    validation_data):  # pylint: disable=unused-argument
    max_distance = 5
    for _ in range(epochs):
      for _ in range(steps_per_epoch):
        pairs, labels = next(batch_gen)

        # Make sure all the pairs are coming from the same trajectory / env
        # and that they are compatible with the given labels.
        batch_x1 = pairs[0]
        batch_x2 = pairs[1]
        batch_size = batch_x1.shape[0]
        assert batch_x2.shape[0] == batch_size
        for k in range(batch_size):
          x1 = batch_x1[k]
          x2 = batch_x2[k]
          env_x1, trajectory_x1, step_x1 = x1
          env_x2, trajectory_x2, step_x2 = x2
          self.assertEqual(env_x1, env_x2)
          self.assertEqual(trajectory_x1, trajectory_x2)

          distance = abs(step_x2 - step_x1)
          if labels[k][1]:
            self.assertLessEqual(distance, max_distance)
          else:
            self.assertGreater(distance, max_distance)

  def test_r_network_training(self):
    r_model = self
    r_trainer = r_network_training.RNetworkTrainer(
        r_model,
        observation_history_size=10000,
        training_interval=20000)

    # Every observation is a vector of dimension 3:
    # (worker_id, trajectory_id, step_num)
    feed_count = 20000
    env_count = 8
    worker_id = list(range(env_count))
    trajectory_id = [0] * env_count
    step_idx = [0] * env_count
    proba_done = 0.01
    for _ in range(feed_count):
      # Observations: size = env_count x 3
      observations = np.stack([worker_id, trajectory_id, step_idx])
      observations = np.transpose(observations)
      dones = np.random.choice([True, False], size=env_count,
                               p=[proba_done, 1.0 - proba_done])
      r_trainer.on_new_observation(observations, None, dones, None)

      step_idx = [s + 1 for s in step_idx]

      # Update the trajectory index and the environment step.
      for k in range(env_count):
        if dones[k]:
          step_idx[k] = 0
          trajectory_id[k] += 1


if __name__ == '__main__':
  absltest.main()
