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

r"""Main file for training policies.

Many hyperparameters need to be passed through gin flags.
Consider using scripts/launcher_script.py to invoke train_policy with the
right hyperparameters and flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

from absl import flags
from episodic_curiosity import env_factory
from episodic_curiosity import eval_policy
from episodic_curiosity import utils
from third_party.baselines import logger
from third_party.baselines.ppo2 import policies
from third_party.baselines.ppo2 import ppo2
import gin
import tensorflow as tf


flags.DEFINE_string('workdir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'CartPole-v0', 'What environment to run')
flags.DEFINE_string('policy_architecture', 'cnn',
                    'What model architecture to use')
flags.DEFINE_string('r_checkpoint', '', 'Location of the R-network checkpoint')
flags.DEFINE_integer('num_env', 12, 'Number of environment copies to run in '
                     'subprocesses.')
flags.DEFINE_string('dmlab_homepath', '', '')
flags.DEFINE_integer('num_timesteps', 10000000, 'Number of frames to run '
                     'training for.')
flags.DEFINE_string('action_set', '',
                    '(small|nofire|) - which action set to use')
flags.DEFINE_bool('use_curiosity', False,
                  'Whether to enable Pathak\'s curiosity')
flags.DEFINE_bool('random_state_predictor', False,
                  'Whether to use random state predictor for Pathak\'s '
                  'curiosity')
flags.DEFINE_float('curiosity_strength', 0.01,
                   'Strength of the intrinsic reward in Pathak\'s algorithm.')
flags.DEFINE_float('forward_inverse_ratio', 0.2,
                   'Weighting of forward vs inverse loss in Pathak\'s '
                   'algorithm')
flags.DEFINE_float('curiosity_loss_strength', 10,
                   'Weight of the curiosity loss in Pathak\'s algorithm.')


# pylint: disable=g-inconsistent-quotes
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
# pylint: enable=g-inconsistent-quotes

FLAGS = flags.FLAGS


def get_environment(env_name):
  dmlab_prefix = 'dmlab:'
  atari_prefix = 'atari:'
  parkour_prefix = 'parkour:'
  if env_name.startswith(dmlab_prefix):
    level_name = env_name[len(dmlab_prefix):]
    return env_factory.create_environments(
        level_name,
        FLAGS.num_env,
        FLAGS.r_checkpoint,
        FLAGS.dmlab_homepath,
        action_set=FLAGS.action_set,
        r_network_weights_store_path=FLAGS.workdir)
  elif env_name.startswith(atari_prefix):
    level_name = env_name[len(atari_prefix):]
    return env_factory.create_environments(
        level_name,
        FLAGS.num_env,
        FLAGS.r_checkpoint,
        environment_engine='atari',
        r_network_weights_store_path=FLAGS.workdir)
  if env_name.startswith(parkour_prefix):
    return env_factory.create_environments(
        env_name[len(parkour_prefix):],
        FLAGS.num_env,
        FLAGS.r_checkpoint,
        environment_engine='parkour',
        r_network_weights_store_path=FLAGS.workdir)
  raise ValueError('Unknown environment: {}'.format(env_name))


@gin.configurable
def train(workdir, env_name, num_timesteps,
          nsteps=256,
          nminibatches=4,
          noptepochs=4,
          learning_rate=2.5e-4,
          ent_coef=0.01):
  """Runs PPO training.

  Args:
    workdir: where to store experiment results/logs
    env_name: the name of the environment to run
    num_timesteps: for how many timesteps to run training
    nsteps: Number of consecutive environment steps to use during training.
    nminibatches: Minibatch size.
    noptepochs: Number of optimization epochs.
    learning_rate: Initial learning rate.
    ent_coef: Entropy coefficient.
  """
  train_measurements = utils.create_measurement_series(workdir, 'reward_train')
  valid_measurements = utils.create_measurement_series(workdir, 'reward_valid')
  test_measurements = utils.create_measurement_series(workdir, 'reward_test')

  def measurement_callback(unused_eplenmean, eprewmean, global_step_val):
    if train_measurements:
      train_measurements.create_measurement(
          objective_value=eprewmean, step=global_step_val)

  def eval_callback_on_valid(eprewmean, global_step_val):
    if valid_measurements:
      valid_measurements.create_measurement(
          objective_value=eprewmean, step=global_step_val)

  def eval_callback_on_test(eprewmean, global_step_val):
    if test_measurements:
      test_measurements.create_measurement(
          objective_value=eprewmean, step=global_step_val)

  logger_dir = workdir
  logger.configure(logger_dir)

  env, valid_env, test_env = get_environment(env_name)
  is_ant = env_name.startswith('parkour:')

  # Validation metric.
  policy_evaluator_on_valid = eval_policy.PolicyEvaluator(
      valid_env,
      metric_callback=eval_callback_on_valid,
      video_filename=None)

  # Test metric (+ videos).
  video_filename = os.path.join(FLAGS.workdir, 'video')
  policy_evaluator_on_test = eval_policy.PolicyEvaluator(
      test_env,
      metric_callback=eval_callback_on_test,
      video_filename=video_filename,
      grayscale=(env_name.startswith('atari:')))

  # Delay to make sure that all the DMLab environments acquire
  # the GPU resources before TensorFlow acquire the rest of the memory.
  # TODO(damienv): Possibly use allow_grow in a TensorFlow session
  # so that there is no such problem anymore.
  time.sleep(15)

  cloud_sync_callback = lambda: None

  def evaluate_valid_test(model_step_fn, global_step):
    if not is_ant:
      policy_evaluator_on_valid.evaluate(model_step_fn, global_step)
    policy_evaluator_on_test.evaluate(model_step_fn, global_step)

  with tf.Session():
    policy = {'cnn': policies.CnnPolicy,
              'lstm': policies.LstmPolicy,
              'lnlstm': policies.LnLstmPolicy,
              'mlp': policies.MlpPolicy}[FLAGS.policy_architecture]

    # Openai baselines never performs num_timesteps env steps because
    # of the way it samples training data in batches. The number of timesteps
    # is multiplied by 1.1 (hacky) to insure at least num_timesteps are
    # performed.

    ppo2.learn(policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
               lam=0.95, gamma=0.99, noptepochs=noptepochs, log_interval=1,
               ent_coef=ent_coef,
               lr=learning_rate if is_ant else lambda f: f * learning_rate,
               cliprange=0.2 if is_ant else lambda f: f * 0.1,
               total_timesteps=int(num_timesteps * 1.1),
               train_callback=measurement_callback,
               eval_callback=evaluate_valid_test,
               cloud_sync_callback=cloud_sync_callback,
               save_interval=200, workdir=workdir,
               use_curiosity=FLAGS.use_curiosity,
               curiosity_strength=FLAGS.curiosity_strength,
               forward_inverse_ratio=FLAGS.forward_inverse_ratio,
               curiosity_loss_strength=FLAGS.curiosity_loss_strength,
               random_state_predictor=FLAGS.random_state_predictor)
    cloud_sync_callback()
  test_env.close()
  valid_env.close()
  utils.maybe_close_measurements(train_measurements)
  utils.maybe_close_measurements(valid_measurements)
  utils.maybe_close_measurements(test_measurements)




def main(_):
  utils.dump_flags_to_file(os.path.join(FLAGS.workdir, 'flags.txt'))
  tf.logging.set_verbosity(tf.logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_files,
                                      FLAGS.gin_bindings)
  train(FLAGS.workdir, env_name=FLAGS.env_name,
        num_timesteps=FLAGS.num_timesteps)


if __name__ == '__main__':
  tf.app.run()
