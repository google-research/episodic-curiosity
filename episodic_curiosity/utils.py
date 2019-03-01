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

"""A few utilities for episodic curiosity.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import csv
import os
import time
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def get_frame(env_observation, info):
  """Searches for a rendered frame in 'info', fallbacks to the env obs."""
  info_frame = info.get('frame')
  if info_frame is not None:
    return info_frame
  return env_observation


def dump_flags_to_file(filename):
  """Dumps FLAGS to a file."""
  with tf.gfile.Open(filename, 'w') as output:
    output.write('\n'.join([
        '{}={}'.format(flag_name, flag_value)
        for flag_name, flag_value in FLAGS.flag_values_dict().items()
    ]))


class MeasurementsWriter(object):
  """Writes measurements to CSV."""

  def __init__(self, workdir, measurement_name):
    """Initializes a MeasurementsWriter.

    Args:
      workdir: Directory to which the CSV file will be written.
      measurement_name: Name of the measurement.
    """
    filename = os.path.join(workdir, measurement_name + '.csv')
    file_exists = tf.gfile.Exists(filename)
    self._out_file = tf.gfile.Open(filename, mode='a+')
    self._csv_writer = csv.writer(self._out_file)
    if not file_exists:
      self._csv_writer.writerow(['step', measurement_name, 'timestamp_s'])
      self._out_file.flush()
    self._measurement_name = measurement_name
    self._last_flush_time = 0

  def create_measurement(self, objective_value, step):
    """Adds a measurement.

    Args:
      objective_value: Value to report for the given training step.
      step: Training step.
    """
    flush_every_s = 5
    self._csv_writer.writerow(
        [str(step), str(objective_value), str(int(time.time()))])
    if time.time() - self._last_flush_time >= flush_every_s:
      self._last_flush_time = time.time()
      self._out_file.flush()

  def close(self):
    del self._csv_writer
    self._out_file.close()


def create_measurement_series(workdir, label):
  """Creates an object for exporting a per-training-step metric."""
  return MeasurementsWriter(workdir, label)  # pylint:disable=unreachable


def maybe_close_measurements(measurements):
  if isinstance(measurements, MeasurementsWriter):
    measurements.close()


def load_keras_model(path):
  """Loads a keras model from a h5 file path."""
  # pylint:disable=unreachable
  return tf.keras.models.load_model(path, compile=True)
