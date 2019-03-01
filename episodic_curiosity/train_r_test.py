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

"""Tests for dune.rl.episodic_curiosity.train_r."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
from absl.testing import absltest
from episodic_curiosity import constants
from episodic_curiosity import keras_checkpoint
from episodic_curiosity import train_r
import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS


class TrainRTest(absltest.TestCase):

  def setUp(self):
    super(TrainRTest, self).setUp()
    keras.backend.clear_session()

  def test_export_stats_to_xm(self):
    xm_series = train_r.XmSeries(
        loss=mock.MagicMock(),
        acc=mock.MagicMock(),
        val_loss=mock.MagicMock(),
        val_acc=mock.MagicMock())
    self._fit_model_with_callback(train_r.ExportStatsToXm(xm_series))
    for series in xm_series._asdict().values():
      self.assertEqual(series.create_measurement.call_count, 1)

  def test_model_weights_checkpoint(self):
    path = os.path.join(FLAGS.test_tmpdir, 'r_network_weights.{epoch:05d}.h5')
    self._fit_model_with_callback(
        keras_checkpoint.GFileModelCheckpoint(
            path,
            save_summary=True,
            summary=constants.Level('explore_goal_locations_small').asdict(),
            save_weights_only=True,
            period=1))
    self.assertTrue(tf.gfile.Exists(path.format(epoch=1)))
    self.assertTrue(
        tf.gfile.Exists(path.format(epoch=1).replace('h5', 'summary.txt')))

  def test_full_model_checkpoint(self):
    path = os.path.join(FLAGS.test_tmpdir, 'r_network_full.{epoch:05d}.h5')
    self._fit_model_with_callback(
        keras_checkpoint.GFileModelCheckpoint(
            path, save_summary=False, save_weights_only=False, period=1))
    self.assertTrue(tf.gfile.Exists(path.format(epoch=1)))
    self.assertFalse(
        tf.gfile.Exists(path.format(epoch=1).replace('h5', 'summary.txt')))

  def _fit_model_with_callback(self, callback):
    inpt = keras.layers.Input(shape=(1,))
    model = keras.models.Model(inputs=inpt, outputs=keras.layers.Dense(1)(inpt))
    model.compile(
        loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(
        x=np.random.rand(100, 1),
        y=np.random.rand(100, 1),
        validation_data=(np.random.rand(100, 1), np.random.rand(100, 1)),
        callbacks=[callback])


if __name__ == '__main__':
  absltest.main()
