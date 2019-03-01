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

"""Class that represents an episodic memory."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np


@gin.configurable
class EpisodicMemory(object):
  """Episodic memory."""

  def __init__(self,
               observation_shape,
               observation_compare_fn,
               replacement='fifo',
               capacity=200):
    """Creates an episodic memory.

    Args:
      observation_shape: Shape of an observation.
      observation_compare_fn: Function used to measure similarity between
        two observations. This function returns the estimated probability that
        two observations are similar.
      replacement: String to select the behavior when a sample is added
        to the memory when this one is full.
        Can be one of: 'fifo', 'random'.
        'fifo' keeps the last "capacity" samples into the memory.
        'random' results in a geometric distribution of the age of the samples
        present in the memory.
      capacity: Capacity of the episodic memory.

    Raises:
      ValueError: when the replacement scheme is invalid.
    """
    self._capacity = capacity
    self._replacement = replacement
    if self._replacement not in ['fifo', 'random']:
      raise ValueError('Invalid replacement scheme')
    self._observation_shape = observation_shape
    self._observation_compare_fn = observation_compare_fn
    self.reset(False)

  def reset(self, show_stats=True):
    """Resets the memory."""
    if show_stats:
      size = len(self)
      age_histogram, _ = np.histogram(self._memory_age[:size],
                                      10, [0, self._count])
      age_histogram = age_histogram.astype(np.float32)
      age_histogram = age_histogram / np.sum(age_histogram)
      print('Number of samples added in the previous trajectory: {}'.format(
          self._count))
      print('Histogram of sample freshness (old to fresh): {}'.format(
          age_histogram))

    self._count = 0
    # Stores environment observations.
    self._obs_memory = np.zeros([self._capacity] + self._observation_shape)
    # Stores the infos returned by the environment. For debugging and
    # visualization purposes.
    self._info_memory = [None] * self._capacity
    self._memory_age = np.zeros([self._capacity], dtype=np.int32)

  @property
  def capacity(self):
    return self._capacity

  def __len__(self):
    return min(self._count, self._capacity)

  @property
  def info_memory(self):
    return self._info_memory

  def add(self, observation, info):
    """Adds an observation to the memory.

    Args:
      observation: Observation to add to the episodic memory.
      info: Info returned by the environment together with the observation,
            for debugging and visualization purposes.

    Raises:
      ValueError: when the capacity of the memory is exceeded.
    """
    if self._count >= self._capacity:
      if self._replacement == 'random':
        # By using random replacement, the age of elements inside the memory
        # follows a geometric distribution (more fresh samples compared to
        # old samples).
        index = np.random.randint(low=0, high=self._capacity)
      elif self._replacement == 'fifo':
        # In this scheme, only the last self._capacity elements are kept.
        # Samples are replaced using a FIFO scheme (implemented as a circular
        # buffer).
        index = self._count % self._capacity
      else:
        raise ValueError('Invalid replacement scheme')
    else:
      index = self._count

    self._obs_memory[index] = observation
    self._info_memory[index] = info
    self._memory_age[index] = self._count
    self._count += 1

  def similarity(self, observation):
    """Similarity between the input observation and the ones from the memory.

    Args:
      observation: The input observation.

    Returns:
      A numpy array of similarities corresponding to the similarity between
      the input and each of the element in the memory.
    """
    # Make the observation batched with batch_size = self._size before
    # computing the similarities.
    # TODO(damienv): could we avoid replicating the observation ?
    # (with some form of broadcasting).
    size = len(self)
    observation = np.array([observation] * size)
    similarities = self._observation_compare_fn(observation,
                                                self._obs_memory[:size])
    return similarities


@gin.configurable
def similarity_to_memory(observation,
                         episodic_memory,
                         similarity_aggregation='percentile'):
  """Returns the similarity of the observation to the episodic memory.

  Args:
    observation: The observation the agent transitions to.
    episodic_memory: Episodic memory.
    similarity_aggregation: Aggregation method to turn the multiple
        similarities to each observation in the memory into a scalar.

  Returns:
    A scalar corresponding to the similarity to episodic memory. This is
    computed by aggregating the similarities between the new observation
    and every observation in the memory, according to 'similarity_aggregation'.
  """
  # Computes the similarities between the current observation and the past
  # observations in the memory.
  memory_length = len(episodic_memory)
  if memory_length == 0:
    return 0.0
  similarities = episodic_memory.similarity(observation)
  # Implements different surrogate aggregated similarities.
  # TODO(damienv): Implement other types of surrogate aggregated similarities.
  if similarity_aggregation == 'max':
    aggregated = np.max(similarities)
  elif similarity_aggregation == 'nth_largest':
    n = min(10, memory_length)
    aggregated = np.partition(similarities, -n)[-n]
  elif similarity_aggregation == 'percentile':
    percentile = 90
    aggregated = np.percentile(similarities, percentile)
  elif similarity_aggregation == 'relative_count':
    # Number of samples in the memory similar to the input observation.
    count = sum(similarities > 0.5)
    aggregated = float(count) / len(similarities)

  return aggregated
