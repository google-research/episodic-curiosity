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

"""Keras checkpointing using GFile."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import tempfile
import time

from absl import logging
import tensorflow as tf
from tensorflow import keras


# Taken from: dos/ml/lstm/train_util.py, but also supports unformatted strings
# and writing summary files.
class GFileModelCheckpoint(keras.callbacks.ModelCheckpoint):
  """Keras callback to checkpoint model to a gfile location.

  Makes the keras ModelCheckpoint callback compatible with google filesystem
  paths, such as CNS files.
  Models will be saved to tmp_file_path and copied from there to file_path.
  Also writes a summary file with model performance along with the checkpoint.
  """

  def __init__(self,
               file_path,
               save_summary,
               summary = None,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Initializes checkpointer with appropriate filepaths.

    Args:
      file_path: gfile location to save model to. Supports unformatted strings
                 similarly to keras ModelCheckpoint.
      save_summary: Whether we should generate and save a summary file.
      summary: Additional items to write to the summary file.
      *args: positional args passed to the underlying ModelCheckpoint.
      **kwargs: named args passed to the underlying ModelCheckpoint.
    """
    self.save_summary = save_summary
    self.summary = summary
    # We assume that this directory is not used by anybody else, so we uniquify
    # it (a bit overkill, but hey).
    self.tmp_dir = os.path.join(
        tempfile.gettempdir(),
        'tmp_keras_weights_%d_%d' % (int(time.time() * 1e6), id(self)))
    tf.gfile.MakeDirs(self.tmp_dir)
    self.tmp_path = os.path.join(self.tmp_dir, os.path.basename(file_path))
    self.gfile_dir = os.path.dirname(file_path)
    super(GFileModelCheckpoint, self).__init__(self.tmp_path, *args, **kwargs)

  def on_epoch_end(self, epoch, logs = None):
    """At end of epoch, performs the gfile checkpointing."""
    super(GFileModelCheckpoint, self).on_epoch_end(epoch, logs=None)
    if self.epochs_since_last_save == 0:  # ModelCheckpoint just saved
      tmp_dir_contents = tf.gfile.ListDirectory(self.tmp_dir)
      for tmp_weights_filename in tmp_dir_contents:
        src = os.path.join(self.tmp_dir, tmp_weights_filename)
        dst = os.path.join(self.gfile_dir, tmp_weights_filename)
        logging.info('Copying saved keras model weights from %s to %s', src,
                     dst)
        tf.gfile.Copy(src, dst, overwrite=True)
        tf.gfile.Remove(src)
        if self.save_summary:
          merged_summary = {}
          merged_summary.update(self.summary)
          if logs:
            merged_summary.update(logs)
          with tf.gfile.Open(dst.replace('.h5', '.summary.txt'),
                             'w') as summary_file:
            summary_file.write('\n'.join(
                ['{}: {}'.format(k, v) for k, v in merged_summary.items()]))
