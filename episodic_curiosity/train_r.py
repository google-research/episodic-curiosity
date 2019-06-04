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

"""Code for training R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import collections
import copy
import os
import re
from absl import app
from absl import flags
from absl import logging
from episodic_curiosity import constants
from episodic_curiosity import keras_checkpoint
from episodic_curiosity import utils
from episodic_curiosity.constants import Const
from third_party.keras_resnet import models
import gin
import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('dmlab_homepath', '', '')
flags.DEFINE_string(
    'fully_qualified_level', 'explore_goal_locations_small',
    'Level to train on. Name should match fully qualified '
    'names in constants.Const. A fake "merged" level is accepted, meaning '
    'that we use the training data from all levels.')
flags.DEFINE_string(
    'input_r_training_dir', None,
    'Mutually exclusive with --training_data_glob, --input_r_training_dir. '
    'Directory containing the training examples shuffled by shuffle_examples. '
    'See build_r_training_data_glob for the expected format of the subdir '
    'used.')
flags.DEFINE_string(
    'training_data_glob', None,
    'Mutually exclusive with --input_from_experiment, --input_r_training_dir. '
    'Glob that should yield all input training files.')
flags.DEFINE_enum(
    'input_type', 'raw_tfrecords', ['tfrecords', 'sstable', 'raw_tfrecords'],
    "Type of files in --input_r_training_dir. 'raw_tfrecords' "
    "means that we don't expect the input to be shuffled, and "
    'perform shuffling locally, in particular interleaving input '
    'files.')
flags.DEFINE_integer(
    'max_input_env_steps', 2500000,
    'Only consider input examples whose example_index is '
    'lower that this flag.')
flags.DEFINE_bool('trainable_bottom_network', True,
                  'Whether the bottom (embedding) network is trainable.')
flags.DEFINE_bool(
    'use_deep_top_network', True,
    ' If true (default), a deep network will be used for'
    'comparing embeddings. Otherwise, we use a simple'
    'distance metric.')
flags.DEFINE_string(
    'training_episode_length', 'default',
    'Episode length to use for training. This must match an '
    'episode length generated in the input data, see '
    '--episode_length in generate_r_training_data.')
flags.DEFINE_string(
    'validation_episode_length', 'default',
    'Episode length to use for validation. This must match an '
    'episode length generated in the input data, see '
    '--episode_length in generate_r_training_data.')
flags.DEFINE_string('noise_type', '', 'What noise type was applied.')
flags.DEFINE_integer('tv_num_images', 0, 'Number of images shown on TV.')
flags.DEFINE_enum('action_set', '',
                  ['', 'small', 'withidle', 'nofire', 'smallwithback',
                   'defaultwithidle'],
                  'Action set to use.')
flags.DEFINE_integer(
    'max_action_distance', -1,
    'Used for selecting a specific input training set. -1 means we pick the '
    'training set where max_action_distance is not specified')
flags.DEFINE_float('adam_lr', -1,
                   'Learning rate for ADAM. Negative means use the default '
                   'from constants.Const')

flags.DEFINE_integer('dataset_buffer_size', 100000,
                     'Dataset buffer size for shuffling the inputs.')
flags.DEFINE_integer(
    'percent_validation_files', 10,
    'Only used together with --training_data_glob. If '
    'positive, the given percentage of input files will be '
    'reserved for validation')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

XmSeries = collections.namedtuple('XmSeries',
                                  ['loss', 'val_loss', 'acc', 'val_acc'])


def build_r_training_data_glob(level, mixer_seed,
                               episode_length, action_set,
                               noise_type, tv_num_images,
                               max_action_distance):
  assert FLAGS.input_r_training_dir
  subdir = '{}:{}:{}:{}:{}:{}:{}'.format(level.fully_qualified_name, mixer_seed,
                                         episode_length, action_set, noise_type,
                                         tv_num_images, max_action_distance)
  return os.path.join(FLAGS.input_r_training_dir, subdir, 'r_training_data*')


def parse_example(serialized_example):
  """Parses a single example generated by generate_r_training_data.py."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'x1': tf.FixedLenFeature([], tf.string),
          'x2': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'example_index': tf.FixedLenFeature([], tf.int64),
          'episode_length': tf.FixedLenFeature([], tf.string),
          'global_env_steps_at_episode': tf.FixedLenFeature([], tf.int64),
      })

  def decode(encoded):
    return tf.reshape(
        tf.image.decode_png(encoded), shape=Const.OBSERVATION_SHAPE)

  return {
      'x1': decode(features['x1']),
      'x2': decode(features['x2']),
      'example_index': features['example_index'],
      'episode_length': features['episode_length'],
      'global_env_steps_at_episode': features['global_env_steps_at_episode'],
  }, tf.one_hot(features['label'], 2)


def filter_examples_fn(example, label,
                       max_input_env_steps,
                       episode_length):
  del label  # unused
  return tf.logical_and(
      tf.less_equal(example['global_env_steps_at_episode'],
                    max_input_env_steps),
      tf.equal(example['episode_length'], episode_length))


def strip_additional_example_info(example,
                                  label):
  # Keys of this dict must match the name of the Inputs of the keras model.
  return ({'x1': example['x1'], 'x2': example['x2']}, label)


class ExportStatsToXm(keras.callbacks.Callback):
  """Keras callback to export model performance as XM measurements."""

  def __init__(self, xm_series):
    super(ExportStatsToXm, self).__init__()
    self.xm_series = xm_series

  def on_epoch_end(self, epoch, logs=None):
    if not logs:
      return
    print(logs)
    for metric, series in self.xm_series._asdict().items():
      if not series:
        continue
      series.create_measurement(logs[metric], epoch)


class RTrainer(object):
  """Class responsible for training R-network."""

  def __init__(self,
               workdir,
               level,
               xm_series = None):
    """Inits a RTrainer.

    Args:
      workdir: Model checkpoints will be saved to this dir.
      level: DMLab level, potentially with extra settings
      xm_series: XM measurement series used to export training stats.
    """
    self.workdir = workdir
    self.xm_series = xm_series
    self.level = level
    self.saved_model_basename_prefix = 'r_network_full'

  def create_dataset(self, filenames, filter_fn):
    """Creates a tf.data.Dataset for the given filenames."""
    logging.info('Creating dataset: %s', filenames)
    assert filenames, 'No input data found'
    if FLAGS.input_type == 'tfrecords':
      dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=10)
      example_parser = parse_example
    elif FLAGS.input_type == 'sstable':
      dataset = tf.data.SSTableDataset(filenames)
      example_parser = lambda key, value: parse_example(value)
    elif FLAGS.input_type == 'raw_tfrecords':
      dataset = tf.data.Dataset.from_tensor_slices(filenames).interleave(
          lambda filename: tf.data.TFRecordDataset(filename).repeat(),
          # Consume data from all input files in parallel.
          cycle_length=len(filenames),
          # We want to get the same number of elements from each input file in
          # the shuffle buffer.
          block_length=1)
      example_parser = parse_example
    else:
      assert False, 'Unknown input type {}'.format(FLAGS.input_type)

    # When the input data is already randomized by shuffle_examples, this
    # randomization is not critical. Still good hygiene to do it here, e.g. so
    # that we get examples in different order when we do multiple passes over
    # the dataset, or to ensure that we don't rely on / overfit to a specific
    # order, e.g. if we would tune hyperparameters.
    # When the input data is not randomized (input_type=raw_sstable), this is
    # critical.
    ds = dataset.map(
        example_parser, num_parallel_calls=5).filter(filter_fn).map(
            strip_additional_example_info).shuffle(
                buffer_size=FLAGS.dataset_buffer_size).batch(
                    Const.BATCH_SIZE).prefetch(100).repeat()
    return ds

  def find_existing_model_path(self, glob):
    """Returns the latest saved model, if it exists."""
    filenames = tf.gfile.Glob(glob)
    if not filenames:
      return None
    return max(filenames)

  def get_epoch(self, filename):
    match = re.search(r'\.(?P<shard>\d{5})\.h5', filename)
    if not match:
      raise ValueError('Did not find epoch in filename ' + filename)
    return int(match.group('shard').lstrip('0'))

  def create_model(self):
    """Creates the keras model to train, either a new one, or from disk."""
    saved_model_filename = self.find_existing_model_path(
        os.path.join(self.workdir, self.saved_model_basename_prefix + '*'))
    if saved_model_filename:
      epoch = self.get_epoch(saved_model_filename)
      logging.info('Loading existing model from path %s (initial epoch=%d)',
                   saved_model_filename, epoch)
      return utils.load_keras_model(saved_model_filename), epoch
    logging.info('Did not find existing model. Starting with a randomly '
                 'initialized model.')
    input_shape = (
        Const.OBSERVATION_HEIGHT,
        Const.OBSERVATION_WIDTH,
        Const.OBSERVATION_CHANNELS,
    )
    model, _, _ = models.ResnetBuilder.build_siamese_resnet_18(
        input_shape,
        use_deep_top_network=FLAGS.use_deep_top_network,
        trainable_bottom_network=FLAGS.trainable_bottom_network)
    adam_params = copy.copy(Const.ADAM_PARAMS)
    if FLAGS.adam_lr > 0:
      adam_params['lr'] = FLAGS.adam_lr
    adam = keras.optimizers.Adam(**adam_params)
    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model, 0

  def train(self):
    """Launch training."""
    logging.info('Started training!')
    model, initial_epoch = self.create_model()
    print('Model we train:')
    model.summary()
    checkpoint_summary = copy.copy(self.level.asdict())
    checkpoint_summary.update(
        action_set=FLAGS.action_set,
        episode_length=FLAGS.training_episode_length,
        max_input_env_steps=FLAGS.max_input_env_steps)
    # Keep saving the weights only (even though not technically needed since we
    # save full model below), so that stay compatible with downstream code.
    weights_checkpoint_cb = keras_checkpoint.GFileModelCheckpoint(
        os.path.join(self.workdir, 'r_network_weights.{epoch:05d}.h5'),
        save_summary=True,
        summary=checkpoint_summary,
        save_weights_only=True,
        period=Const.STORE_CHECKPOINT_EVERY_N_EPOCHS)
    model_checkpoint_cb = keras_checkpoint.GFileModelCheckpoint(
        os.path.join(self.workdir,
                     '%s.{epoch:05d}.h5' % self.saved_model_basename_prefix),
        save_summary=False,
        save_weights_only=False,
        # Keras does not store the dataset/iterator state, so training will
        # start from the beginning of the dataset. To make sure this is not a
        # problem (e.g. too many epochs training on the beginning of the dataset
        # because of restarts, leading to overfitting), we ensure that we've
        # iterated at least once through the dataset in the worst case before
        # dumping the full model checkpoint.
        # The dataset contains at most 800k examples, and we consume 6400 of
        # those per epoch, so one iteration over the dataset is at most 123
        # epochs.
        period=123)
    callbacks = [weights_checkpoint_cb, model_checkpoint_cb]
    if self.xm_series:
      callbacks.append(ExportStatsToXm(self.xm_series))

    assert (bool(FLAGS.input_r_training_dir) + bool(FLAGS.training_data_glob)
            == 1), (
                'Exactly one of --input_r_training_dir, training_data_glob, '
                'input_from_experiment should be set.')

    if FLAGS.training_data_glob:
      filenames = tf.gfile.Glob(FLAGS.training_data_glob)
      filenames.sort()
      logging.info('Files with glob %s: %s', FLAGS.training_data_glob,
                   filenames)
      training_start_index = int(
          FLAGS.percent_validation_files * len(filenames) / 100.)
      training_filenames = filenames[training_start_index:]
      validation_filenames = filenames[:training_start_index]
      assert validation_filenames, (
          'No validation filename. Increase --percent_validation_files.')
    elif FLAGS.input_r_training_dir:
      training_glob = build_r_training_data_glob(
          self.level, Const.MIXER_SEEDS[constants.SplitType.R_TRAINING],
          FLAGS.training_episode_length, FLAGS.action_set, FLAGS.noise_type,
          FLAGS.tv_num_images, FLAGS.max_action_distance)
      logging.info('Looking for training files with glob: %s', training_glob)
      training_filenames = tf.gfile.Glob(training_glob)
      validation_glob = build_r_training_data_glob(
          self.level, Const.MIXER_SEEDS[constants.SplitType.VALIDATION],
          FLAGS.validation_episode_length, FLAGS.action_set, FLAGS.noise_type,
          FLAGS.tv_num_images, FLAGS.max_action_distance)
      logging.info('Looking for validation files with glob: %s',
                   validation_glob)
      validation_filenames = tf.gfile.Glob(validation_glob)

    # pylint: disable=g-long-lambda
    training_filter_fn = lambda example, label: filter_examples_fn(
        example, label, FLAGS.max_input_env_steps,
        FLAGS.training_episode_length)

    training_dataset = self.create_dataset(training_filenames,
                                           training_filter_fn)

    validation_filter_fn = lambda example, label: filter_examples_fn(
        example, label, FLAGS.max_input_env_steps,
        FLAGS.validation_episode_length)
    # pylint: enable=g-long-lambda

    validation_dataset = self.create_dataset(validation_filenames,
                                             validation_filter_fn)

    model.fit(
        training_dataset.make_one_shot_iterator(),
        steps_per_epoch=Const.DUMP_AFTER_BATCHES,
        epochs=Const.EDGE_MAX_EPOCHS,
        validation_data=validation_dataset.make_one_shot_iterator(),
        validation_steps=100,
        callbacks=callbacks,
        initial_epoch=initial_epoch)
    logging.info('Done training!')


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.workdir):
    tf.gfile.MakeDirs(FLAGS.workdir)

  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      FLAGS.gin_bindings)

  series_dict = {}
  for metric in XmSeries._fields:
    series_dict[metric] = utils.create_measurement_series(FLAGS.workdir, metric)
  xm_series = XmSeries(**series_dict)  # type: ignore

  if FLAGS.fully_qualified_level == 'merged':
    # Use a fake level when merged training data is used. This corresponds to
    # the directory used by shuffle_examples.cc to write the merged dataset.
    level = constants.Level('merged')
  else:
    level = Const.find_level(FLAGS.fully_qualified_level)
  r_trainer = RTrainer(workdir=FLAGS.workdir, level=level, xm_series=xm_series)
  r_trainer.train()
  for series in xm_series._asdict().values():
    if not series:
      continue
    utils.maybe_close_measurements(series)


if __name__ == '__main__':
  app.run(main)
