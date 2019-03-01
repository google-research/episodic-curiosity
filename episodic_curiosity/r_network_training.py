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

"""Set of functions used to train a R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import random

from episodic_curiosity import keras_checkpoint
from episodic_curiosity.constants import Const
import gin
import numpy as np
from tensorflow import keras


def generate_positive_example(buffer_position,
                              next_buffer_position):
  """Generates a close enough pair of states."""
  first = buffer_position
  second = next_buffer_position

  # Make R-network symmetric.
  # Works for DMLab (navigation task), the symmetry assumption might not be
  # valid for all the environments.
  if random.random() < 0.5:
    first, second = second, first
  return first, second


def generate_negative_example(buffer_position,
                              len_episode_buffer,
                              max_action_distance):
  """Generates a far enough pair of states."""
  assert buffer_position < len_episode_buffer
  # Defines the interval that must be excluded from the sampling.
  time_interval = (Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance)
  min_index = max(buffer_position - time_interval, 0)
  max_index = min(buffer_position + time_interval + 1, len_episode_buffer)

  # Randomly select an index outside the interval.
  effective_length = len_episode_buffer - (max_index - min_index)
  range_max = effective_length - 1
  if range_max <= 0:
    return buffer_position, None
  index = random.randint(0, range_max)
  if index >= min_index:
    index = max_index + (index - min_index)
  return buffer_position, index


def compute_next_buffer_position(buffer_position,
                                 positive_example_candidate,
                                 max_action_distance,
                                 mode):
  """Computes the buffer position for the next training example."""
  if mode == 'v3_affect_num_training_examples_overlap':
    # This version was initially not intended (changing max_action_distance
    # affects the number of training examples, and we can also get overlap
    # across generated examples), but we have it because it produces good
    # results (reward at ~40 according to raveman@ on 2018-10-03).
    # R-nets /cns/vz-d/home/dune/episodic_curiosity/raphaelm_train_r_mad2_4 were
    # generated with this version (the flag was set
    # v1_affect_num_training_examples, but it referred to a "buggy" version of
    # v1 that is reproduced here with that v3).
    return buffer_position + random.randint(1, max_action_distance) + 1
  if mode == 'v1_affect_num_training_examples':
    return positive_example_candidate + 1
  if mode == 'v2_fixed_num_training_examples':
    # Produces the ablation study in the paper submitted to ICLR'19
    # (https://openreview.net/forum?id=SkeK3s0qKQ), section S4.1.
    return buffer_position + random.randint(1, 5) + 1


def create_training_data_from_episode_buffer_v4(episode_buffer,
                                                max_action_distance,
                                                avg_num_examples_per_env_step):
  """Sampling of positive/negative examples without using stride logic."""
  num_examples = int(avg_num_examples_per_env_step * len(episode_buffer))
  num_examples_per_class = num_examples // 2
  # We first generate positive pairs, and then sample from them (ensuring that
  # we don't select twice exactly the same pair (i,i+j)).
  positive_pair_candidates = []
  for first in range(len(episode_buffer)):
    for j in range(1, max_action_distance + 1):
      second = first + j
      if second >= len(episode_buffer):
        continue
      positive_pair_candidates.append(
          (first, second) if random.random() > 0.5 else (second, first))
  assert len(positive_pair_candidates) >= num_examples_per_class
  positive_pairs = random.sample(positive_pair_candidates,
                                 num_examples_per_class)

  # Generate negative pairs.
  num_negative_candidates = len(episode_buffer) * (
      len(episode_buffer) -
      2 * Const.NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance) / 2
  # Make sure we have enough negative examples to sample from (with some
  # headroom). If that does not happen (meaning very short episode buffer given
  # current values of negative_sample_multiplier, max_action_distance), don't
  # generate any training example.
  if num_negative_candidates < 2 * num_examples_per_class:
    return [], [], []
  negative_pairs = set()
  while len(negative_pairs) < num_examples_per_class:
    i = random.randint(0, len(episode_buffer) - 1)
    j = generate_negative_example(
        i, len(episode_buffer), max_action_distance)[1]
    # Checking this is not strictly required, because it should happen
    # infrequently with current parameter values.
    # We still check for it for the symmetry with the positive example case.
    if (i, j) not in negative_pairs and (j, i) not in negative_pairs:
      negative_pairs.add((i, j))

  x1 = []
  x2 = []
  labels = []
  for i, j in positive_pairs:
    x1.append(episode_buffer[i])
    x2.append(episode_buffer[j])
    labels.append(1)
  for i, j in negative_pairs:
    x1.append(episode_buffer[i])
    x2.append(episode_buffer[j])
    labels.append(0)
  return x1, x2, labels


def create_training_data_from_episode_buffer_v123(episode_buffer,
                                                  max_action_distance,
                                                  mode):
  """Samples intervals and forms pairs."""
  first_second_label = []
  buffer_position = 0
  while True:
    positive_example_candidate = (
        buffer_position + random.randint(1, max_action_distance))
    next_buffer_position = compute_next_buffer_position(
        buffer_position, positive_example_candidate,
        max_action_distance, mode)

    if (next_buffer_position >= len(episode_buffer) or
        positive_example_candidate >= len(episode_buffer)):
      break
    label = random.randint(0, 1)
    if label:
      first, second = generate_positive_example(buffer_position,
                                                positive_example_candidate)
    else:
      first, second = generate_negative_example(buffer_position,
                                                len(episode_buffer),
                                                max_action_distance)
    if first is None or second is None:
      break
    first_second_label.append((first, second, label))
    buffer_position = next_buffer_position
  x1 = []
  x2 = []
  labels = []
  for first, second, label in first_second_label:
    x1.append(episode_buffer[first])
    x2.append(episode_buffer[second])
    labels.append(label)
  return x1, x2, labels


@gin.configurable
class RNetworkTrainer(object):
  """Train a R network in an online way."""

  def __init__(self,
               r_model,
               observation_history_size=20000,
               training_interval=20000,
               num_epochs=6,
               checkpoint_dir=None):
    # The training interval is assumed to be the same as the history size
    # for invalid negative values.
    if training_interval < 0:
      training_interval = observation_history_size

    self._r_model = r_model
    self._training_interval = training_interval
    self._batch_size = 64
    self._num_epochs = num_epochs

    # Keeps track of the last N observations.
    # Those are used to train the R network in an online way.
    self._fifo_observations = [None] * observation_history_size
    self._fifo_dones = [None] * observation_history_size
    self._fifo_index = 0
    self._fifo_count = 0

    # Used to save checkpoints.
    self._current_epoch = 0
    self._checkpointer = None
    if checkpoint_dir is not None:
      checkpoint_period_in_epochs = self._num_epochs
      self._checkpointer = keras_checkpoint.GFileModelCheckpoint(
          os.path.join(checkpoint_dir, 'r_network_weights.{epoch:05d}.h5'),
          save_summary=False,
          save_weights_only=True,
          period=checkpoint_period_in_epochs)
      self._checkpointer.set_model(self._r_model)

  def on_new_observation(self, observations, unused_rewards, dones, infos):
    """Event triggered when the environments generate a new observation."""
    if len(observations.shape) >= 3 or infos is None or 'frame' not in infos:
      self._fifo_observations[self._fifo_index] = observations
    else:
      # Specific to Parkour (stores velocity, joints as the primary
      # observation).
      self._fifo_observations[self._fifo_index] = infos['frame']
    self._fifo_dones[self._fifo_index] = dones
    self._fifo_index = (
        (self._fifo_index + 1) % len(self._fifo_observations))
    self._fifo_count += 1

    if (self._fifo_count > 0 and
        self._fifo_count % self._training_interval == 0):
      print('Training the R-network after: {}'.format(
          self._fifo_count))
      history_observations, history_dones = self._get_flatten_history()
      self.train(history_observations, history_dones)

  def _get_flatten_history(self):
    """Convert the history given as a circular fifo to a linear array."""
    if self._fifo_count < len(self._fifo_observations):
      return (self._fifo_observations[:self._fifo_count],
              self._fifo_dones[:self._fifo_count])

    # Reorder the indices.
    history_observations = self._fifo_observations[self._fifo_index:]
    history_observations.extend(self._fifo_observations[:self._fifo_index])
    history_dones = self._fifo_dones[self._fifo_index:]
    history_dones.extend(self._fifo_dones[:self._fifo_index])
    return history_observations, history_dones

  def _split_history(self, observations, dones):
    """Returns some individual trajectories."""
    if len(observations) == 0:  # pylint: disable=g-explicit-length-test
      return []

    # Number of environments that generated "observations",
    # and total number of steps.
    nenvs = len(dones[0])
    nsteps = len(dones)

    # Starting index of the current trajectory.
    start_index = [0] * nenvs

    trajectories = []
    for k in range(nsteps):
      for n in range(nenvs):
        if dones[k][n] or k == nsteps - 1:
          next_start_index = k + 1
          time_slice = observations[start_index[n]:next_start_index]
          trajectories.append([obs[n] for obs in time_slice])
          start_index[n] = next_start_index

    return trajectories

  def _prepare_data(self, observations, dones):
    """Generate the positive and negative pairs used to train the R network."""
    max_action_distance = 5
    mode = 'v2_fixed_num_training_examples'

    all_x1 = []
    all_x2 = []
    all_labels = []
    trajectories = self._split_history(observations, dones)
    for trajectory in trajectories:
      x1, x2, labels = create_training_data_from_episode_buffer_v123(
          trajectory, max_action_distance, mode)
      all_x1.extend(x1)
      all_x2.extend(x2)
      all_labels.extend(labels)

    return all_x1, all_x2, all_labels

  def _shuffle(self, x1, x2, labels):
    sample_count = len(x1)
    assert len(x2) == sample_count
    assert len(labels) == sample_count
    permutation = np.random.permutation(sample_count)
    x1 = [x1[p] for p in permutation]
    x2 = [x2[p] for p in permutation]
    labels = [labels[p] for p in permutation]
    return x1, x2, labels

  def train(self, history_observations, history_dones):
    """Do one pass of training of the R-network."""
    x1, x2, labels = self._prepare_data(history_observations, history_dones)
    x1, x2, labels = self._shuffle(x1, x2, labels)

    # Split between train and validation data.
    n = len(x1)
    train_count = (95 * n) // 100
    x1_train, x2_train, labels_train = (
        x1[:train_count], x2[:train_count], labels[:train_count])
    x1_valid, x2_valid, labels_valid = (
        x1[train_count:], x2[train_count:], labels[train_count:])

    validation_data = ([np.array(x1_valid), np.array(x2_valid)],
                       keras.utils.to_categorical(
                           np.array(labels_valid),
                           num_classes=2))

    # TODO(damienv): model checkpointing of R-network should be done
    # in ppo2.py, check whether it includes the R-network part.
    # Otherwise, we would train but never saves the model.
    self._r_model.fit_generator(
        self._generate_batch(x1_train, x2_train, labels_train),
        steps_per_epoch=train_count // self._batch_size,
        epochs=self._num_epochs,
        validation_data=validation_data)

    # Note: the same could possibly be achieved using parameters "callback",
    # "initial_epoch", "epochs" in fit_generator. However, this is not really
    # clear how this initial epoch is supposed to work.
    # TODO(damienv): change it to use callbacks of fit_generator.
    for _ in range(self._num_epochs):
      self._current_epoch += 1
      if self._checkpointer is not None:
        self._checkpointer.on_epoch_end(self._current_epoch)

  def _generate_batch(self, x1, x2, labels):
    """Generate batches of data used to train the R network."""
    while True:
      # Train for one epoch.
      sample_count = len(x1)
      number_of_batches = sample_count // self._batch_size
      for batch_index in range(number_of_batches):
        from_index = batch_index * self._batch_size
        to_index = (batch_index + 1) * self._batch_size
        yield ([np.array(x1[from_index:to_index]),
                np.array(x2[from_index:to_index])],
               keras.utils.to_categorical(
                   np.array(labels[from_index:to_index]),
                   num_classes=2))

      # After each epoch, shuffle the data.
      x1, x2, labels = self._shuffle(x1, x2, labels)
