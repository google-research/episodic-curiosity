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

"""Tests for generate_r_training_data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from episodic_curiosity.environments import fake_gym_env
from episodic_curiosity.generate_r_training_data import generate_random_episode_buffer


class TestGenerateRTrainingData(absltest.TestCase):

  def setUp(self):
    # Fake Environment with an infinite length episode.
    self.env = fake_gym_env.FakeGymEnv()

  def test_generate_random_episode(self):
    for _ in range(2):
      episode_buffer = generate_random_episode_buffer(self.env)
      self.assertEqual(len(episode_buffer), self.env.EPISODE_LENGTH)
      self.assertTupleEqual(episode_buffer[0][0].shape,
                            self.env.OBSERVATION_SHAPE)


if __name__ == '__main__':
  absltest.main()
