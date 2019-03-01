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

"""Training data generation for R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import io
import os
import subprocess
import sys
import tempfile
from absl import app
from absl import flags
from absl import logging
import concurrent.futures
from episodic_curiosity import constants
from episodic_curiosity import env_factory
from episodic_curiosity import r_network_training
from episodic_curiosity.constants import Const
from episodic_curiosity.environments import dmlab_utils
import gin
import numpy as np
import png
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir', None,
    'Root directory for writing training data for the R network.')
flags.DEFINE_string('dmlab_homepath', '', '')
flags.DEFINE_string(
    'fully_qualified_level', 'explore_goal_locations_large',
    'Level to train on. Name should match fully qualified '
    'names in constants.Const')
flags.DEFINE_integer(
    'total_env_steps', 2500000,
    'Total number of environment steps (across all tasks). When '
    'max_action_distance=5, we need (on average) 4 env steps to generate a '
    'single training example.')
flags.DEFINE_integer(
    'num_examples_per_output_shard', 4000,
    'Number of examples for each output file shard. When training the R'
    'network, all shards will be read in parallel, which contributes to '
    'shuffling the data. Be mindful that changing this flag can affect the '
    'quality of the shuffling.')
flags.DEFINE_integer(
    'num_workers', 30,
    'Number of parallel subprocesses to use for running DMLab. '
    'If num_workers==1, we run DMLab directly without spawning a sub-process.')
flags.DEFINE_integer(
    'num_tasks', 160,
    'Number of parallel subprocesses to use for running DMLab. '
    'If num_workers==1, we run DMLab directly without spawning a sub-process.')
flags.DEFINE_integer(
    'task_id', -1,
    'Task ID. If negative and num_workers>1, this binary will spawn '
    '--num_workers subprocess workers. If non-negative or num_workers>1, '
    'the binary is in worker mode: it generates and stores training example '
    'for the R network using DMLab environment. If num_workers==1, there is '
    'no master/worker concept, the main binary runs DMLab directly.')

flags.DEFINE_enum('split', 'R_TRAINING', [s.name for s in constants.SplitType],
                  'Split for which we generate the trajectories')
flags.DEFINE_enum(
    'episode_length',
    'default',
    [
        # Default level episode length (60, 90, 120 depending on level)
        'default',
        '180',
        # This corresponds to ~10k actions (initial R-network training that is
        # known to work well).
        '600',
    ],
    'Length of the episodes.')
flags.DEFINE_enum('action_set', '',
                  ['', 'small', 'nofire', 'withidle', 'defaultwithidle',
                   'smallwithback'],
                  'Action set to use when generating R training data.')
flags.DEFINE_integer(
    'max_action_distance', 5,
    'Parameter that controls the maximum number of env steps '
    'difference between two positive example frames generated '
    'for training the R network')
flags.DEFINE_enum(
    'max_action_distance_mode', 'v1_affect_num_training_examples', [
        'v1_affect_num_training_examples',
        'v2_fixed_num_training_examples',
        'v3_affect_num_training_examples_overlap',
        'v4_no_strides',
    ], 'Controls how max_action_distance affects the number of '
    'training examples produced, given the same number '
    'of environment steps.')
flags.DEFINE_float(
    'avg_num_examples_per_env_step', 1,
    'Has an effect only when max_action_distance_mode=v4_no_strides. '
    'This is the average number of examples we produce for each input '
    'environment step.')
# pylint: disable=g-inconsistent-quotes
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
# pylint: enable=g-inconsistent-quotes


def generate_random_episode_buffer(env):
  """Generates random continuous gameplay."""
  observation = env.reset()
  episode_buffer = []
  while True:
    action = env.action_space.sample()
    observation, _, done, info = env.step(action)
    if done:
      break
    episode_buffer.append((observation, info))
  return episode_buffer


def create_training_data_from_episode_buffer(episode_buffer):
  """Samples intervals and forms pairs."""
  if FLAGS.max_action_distance_mode == 'v4_no_strides':
    return r_network_training.create_training_data_from_episode_buffer_v4(
        episode_buffer,
        FLAGS.max_action_distance,
        FLAGS.avg_num_examples_per_env_step)
  else:
    return r_network_training.create_training_data_from_episode_buffer_v123(
        episode_buffer,
        FLAGS.max_action_distance,
        FLAGS.max_action_distance_mode)


def make_seed():
  return 123 + FLAGS.task_id


def create_env(level):
  """Creates a DMLab environment for generating R training data."""
  main_observation = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'
  env_settings = dmlab_utils.create_env_settings(
      level.dmlab_level_name,
      homepath=FLAGS.dmlab_homepath,
      width=Const.OBSERVATION_WIDTH,
      height=Const.OBSERVATION_HEIGHT,
      seed=make_seed(),
      main_observation=main_observation)
  env_settings.update(level.extra_env_settings)
  env_settings.update(
      mixerSeed=Const.MIXER_SEEDS[constants.SplitType[FLAGS.split]])
  if FLAGS.episode_length != 'default':
    env_settings['episodeLengthSeconds'] = float(FLAGS.episode_length)

  # Saves those parameters since env_settings can be modified by
  # the DMLabWrapper.
  seed = env_settings['seed']
  mixer_seed = env_settings['mixerSeed']

  return dmlab_utils.DMLabWrapper(
      'dmlab',
      env_settings,
      action_set=env_factory.get_action_set(FLAGS.action_set),
      action_repeat=Const.ACTION_REPEAT,
      main_observation=main_observation
  ), seed, mixer_seed


def add_image_feature(example, feature_name,
                      image):
  """Adds an image feature to the tf Example."""
  byte_buffer = io.BytesIO()
  png.from_array(image, 'RGB').save(byte_buffer)
  example.features.feature[feature_name].bytes_list.value.append(
      byte_buffer.getvalue())


def add_integer_feature(example, feature_name,
                        integer):
  """Adds an integer feature to the tf Example."""
  example.features.feature[feature_name].int64_list.value.append(integer)


def add_float_feature(example, feature_name,
                      value):
  """Adds an float feature to the tf Example."""
  if isinstance(value, float):
    example.features.feature[feature_name].float_list.value.append(value)
  else:
    example.features.feature[feature_name].float_list.value.extend(
        value.flatten())


def add_bytes_feature(example, feature_name,
                      feature_value):
  """Adds a bytes feature to the tf Example."""
  example.features.feature[feature_name].bytes_list.value.append(feature_value)


def get_sharded_filename(task_id, shard_id, tmp=False):
  return os.path.join(
      FLAGS.workdir, '{}r_training_data_{}_{}.tfrecords'.format(
          'tmp_' if tmp else '', task_id, shard_id))


def generate_r_training_data():
  """Runs R training data generation."""
  env, seed, mixer_seed = create_env(
      Const.find_level(FLAGS.fully_qualified_level))
  total_examples = 0
  env_steps = 0
  examples_in_shard = 0
  episode = 0
  shard = 0
  writer = None
  max_task_env_steps = FLAGS.total_env_steps // FLAGS.num_tasks
  logging.info('Task %d will run %d env steps',
               FLAGS.task_id, max_task_env_steps)
  while env_steps < max_task_env_steps:
    if (examples_in_shard >= FLAGS.num_examples_per_output_shard or
        writer is None):
      if writer:
        writer.close()
        tf.gfile.Rename(
            get_sharded_filename(FLAGS.task_id, shard, tmp=True),
            get_sharded_filename(FLAGS.task_id, shard, tmp=False),
            overwrite=True)
        shard += 1
      logging.info('Starting shard %d for task %d', shard, FLAGS.task_id)
      writer = tf.python_io.TFRecordWriter(
          get_sharded_filename(FLAGS.task_id, shard, tmp=True))
      examples_in_shard = 0
    episode_buffer = generate_random_episode_buffer(env)
    start_position = None
    if episode_buffer:
      start_position = episode_buffer[0][1]['position']
    x1, x2, labels = create_training_data_from_episode_buffer(episode_buffer)
    for example_index_in_episode, features in enumerate(zip(x1, x2, labels)):
      xx1, xx2, label = features
      example = tf.train.Example()
      add_image_feature(example, 'x1', xx1[0])
      add_image_feature(example, 'x2', xx2[0])
      add_integer_feature(example, 'label', label)
      add_integer_feature(example, 'seed', seed)
      add_integer_feature(example, 'mixer_seed', mixer_seed)
      example.features.feature['fully_qualified_level'].bytes_list.value.append(
          FLAGS.fully_qualified_level.encode('utf-8'))
      # This is the episode index for the current generation task.
      add_integer_feature(example, 'episode', episode)
      # This is the index of the example in the current episode.
      add_integer_feature(example, 'example_index_in_episode',
                          example_index_in_episode)
      # This is the index of the example for the current generation task.
      add_integer_feature(example, 'example_index',
                          total_examples + example_index_in_episode)
      add_integer_feature(example, 'env_steps_at_episode', env_steps)
      # Filtering by this feature leads to a dataset that is equivalent to one
      # generated with the given number of total environment steps (modulo
      # boundary effects).
      add_integer_feature(example, 'global_env_steps_at_episode',
                          env_steps * FLAGS.num_tasks)
      add_bytes_feature(example, 'x1/maze_layout',
                        xx1[1]['maze_layout'].encode('ascii'))
      # For now, x1 and x2 are in the same maze. However, this may not be true
      # in the near future.
      add_bytes_feature(example, 'x2/maze_layout',
                        xx2[1]['maze_layout'].encode('ascii'))
      add_float_feature(example, 'x1/position', xx1[1]['position'])
      add_float_feature(example, 'x2/position', xx2[1]['position'])
      add_float_feature(example, 'x1/dist_from_start',
                        np.linalg.norm(xx1[1]['position'] - start_position))
      add_float_feature(example, 'x2/dist_from_start',
                        np.linalg.norm(xx2[1]['position'] - start_position))
      add_float_feature(example, 'x1/rotation', xx1[1]['rotation'])
      add_float_feature(example, 'x2/rotation', xx2[1]['rotation'])
      add_float_feature(example, 'x1/velocity', xx1[1]['velocity'])
      add_float_feature(example, 'x2/velocity', xx2[1]['velocity'])
      example.features.feature['episode_length'].bytes_list.value.append(
          FLAGS.episode_length.encode('utf-8'))
      example.features.feature['action_set'].bytes_list.value.append(
          FLAGS.action_set.encode('utf-8'))
      try:
        noise_type = gin.query_parameter('DMLabWrapper.noise_type')
      except ValueError:
        noise_type = ''
      add_bytes_feature(example, 'noise_type', noise_type.encode('utf-8'))
      try:
        tv_num_images = int(gin.query_parameter('DMLabWrapper.tv_num_images'))
      except ValueError:
        tv_num_images = 0
      add_integer_feature(example, 'tv_num_images', tv_num_images)
      add_integer_feature(example, 'max_action_distance',
                          FLAGS.max_action_distance)
      writer.write(example.SerializeToString())
    env_steps += len(episode_buffer)
    examples_in_shard += len(x1)
    total_examples += len(x1)
    episode += 1
  writer.close()  # type: ignore
  tf.gfile.Rename(
      get_sharded_filename(FLAGS.task_id, shard, tmp=True),
      get_sharded_filename(FLAGS.task_id, shard, tmp=False),
      overwrite=True)


def run_env_as_sub_process(task_id):
  """Spawns a subprocess that runs a DMLab environment."""
  flags_dict = {f: FLAGS[f].value for f in FLAGS if FLAGS[f].present}
  call = [
      'python',
      '-m',
      'episodic_curiosity.generate_r_training_data',
      '--task_id=' + str(task_id),
  ] + ['--{}={}'.format(k, v) for k, v in flags_dict.items()]
  logging.info('Starting task %d with: %s"', task_id, call)
  output = subprocess.check_output(call)
  logging.info('Received output from subprocess:\n %s', output)




def main(unused_argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      FLAGS.gin_bindings)
  if FLAGS.num_workers > 1 and FLAGS.task_id >= 0:
    generate_r_training_data()
    return

  if tf.gfile.Exists(FLAGS.workdir):
    # Start on a clean state. It is not necessarily safe to restart the code
    # above when some tfrecord files already exist.
    tf.gfile.DeleteRecursively(FLAGS.workdir)
  tf.gfile.MakeDirs(FLAGS.workdir)

  if FLAGS.num_workers == 1:
    generate_r_training_data()
  else:
    logging.info('running %d workers', FLAGS.num_workers)
    with concurrent.futures.ThreadPoolExecutor(FLAGS.num_workers) as executor:
      successful_tasks = 0
      failed_tasks = 0
      next_task_id = 0
      while successful_tasks < FLAGS.num_tasks:
        assert failed_tasks < 40, (
            'Too many failures ({} failures, {} successes)'.format(
                failed_tasks, successful_tasks))
        remaining_tasks = FLAGS.num_tasks - successful_tasks
        logging.info('Scheduling %d remaining tasks', remaining_tasks)
        results = []
        for _ in range(remaining_tasks):
          results.append(executor.submit(run_env_as_sub_process, next_task_id))
          next_task_id += 1
        for result in results:
          if result.exception() is None:
            successful_tasks += 1
            logging.info('One successful task returned (total successful: %d).',
                         successful_tasks)
          else:
            failed_tasks += 1
            logging.info('Failed task (%d): %s',
                         failed_tasks, result.exception())


if __name__ == '__main__':
  app.run(main)
