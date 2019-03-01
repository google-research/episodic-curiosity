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

"""Test of episodic_memory.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from episodic_curiosity import episodic_memory
import numpy as np
import tensorflow as tf


def embedding_similarity(x1, x2):
  assert x1.shape[0] == x2.shape[0]
  epsilon = 1e-6

  # Inner product between the embeddings in x1
  # and the embeddings in x2.
  s = np.sum(x1 * x2, axis=-1)

  s /= np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + epsilon
  return 0.5 * (s + 1.0)


class EpisodicMemoryTest(tf.test.TestCase):

  def RunTest(self, memory, observation_shape, add_count):
    expected_size = min(add_count, memory.capacity)

    for _ in range(add_count):
      observation = np.random.normal(size=observation_shape)
      memory.add(observation, dict())
    self.assertEqual(expected_size, len(memory))

    current_observation = np.random.normal(size=observation_shape)
    similarities = memory.similarity(current_observation)
    self.assertEqual(expected_size, len(similarities))
    self.assertAllLessEqual(similarities, 1.0)
    self.assertAllGreaterEqual(similarities, 0.0)

  def testEpisodicMemory(self):
    observation_shape = [9]
    memory = episodic_memory.EpisodicMemory(
        observation_shape=observation_shape,
        observation_compare_fn=embedding_similarity,
        capacity=150)

    self.RunTest(memory,
                 observation_shape,
                 add_count=100)
    memory.reset()

    self.RunTest(memory,
                 observation_shape,
                 add_count=200)
    memory.reset()


if __name__ == '__main__':
  tf.test.main()
