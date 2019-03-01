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

"""Test of env_factory.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from episodic_curiosity import env_factory
from episodic_curiosity import r_network
from episodic_curiosity.environments import fake_gym_env
from third_party.keras_resnet import models
import numpy as np
import tensorflow as tf
from tensorflow import keras


FLAGS = flags.FLAGS


class EnvFactoryTest(tf.test.TestCase):

  def setUp(self):
    super(EnvFactoryTest, self).setUp()
    keras.backend.clear_session()
    self.weight_path = os.path.join(tf.test.get_temp_dir(), 'weights.h5')
    self.input_shape = fake_gym_env.FakeGymEnv.OBSERVATION_SHAPE
    self.dumped_r_network, _, _ = models.ResnetBuilder.build_siamese_resnet_18(
        self.input_shape)
    self.dumped_r_network.compile(
        loss='categorical_crossentropy', optimizer=keras.optimizers.Adam())
    self.dumped_r_network.save_weights(self.weight_path)

  def testCreateRNetwork(self):
    r_network.RNetwork(self.input_shape, self.weight_path)

  def testCreateAndRunEnvironment(self):
    # pylint: disable=g-long-lambda
    env_factory.create_single_env = (
        lambda level_name, seed, dmlab_homepath, use_monitor, split, action_set:
        fake_gym_env.FakeGymEnv())
    # pylint: enable=g-long-lambda

    env, env_valid, env_test = env_factory.create_environments(
        'explore_object_locations_small', 1, self.weight_path)
    env.reset()
    actions = [0]
    for _ in range(5):
      env.step(actions)
    env.close()
    env_valid.close()
    env_test.close()

  def testRNetworkLazyLoading(self):
    """Tests that the RNetwork weights are lazy loaded."""
    # pylint: disable=g-long-lambda
    rand_obs = lambda: np.random.uniform(low=-0.01, high=0.01,
                                         size=(1,) + self.input_shape)
    obs1 = rand_obs()
    obs2 = rand_obs()
    expected_similarity = self.dumped_r_network.predict([obs1, obs2])
    r_net = r_network.RNetwork(self.input_shape, self.weight_path)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      emb1 = r_net.embed_observation(obs1)
      emb2 = r_net.embed_observation(obs2)
      similarity = r_net.embedding_similarity(emb1, emb2)
      self.assertAlmostEqual(similarity[0], expected_similarity[0, 1])


if __name__ == '__main__':
  tf.test.main()
