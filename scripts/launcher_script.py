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

"""Script that launches policy training with the right hyperparameters.

All specified runs are launched in parallel as subprocesses.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import subprocess

from absl import app
from absl import flags

from episodic_curiosity import constants
import six
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None,
                    'Directory where all experiment results will be stored')
flags.mark_flag_as_required('workdir')

flags.DEFINE_enum(
    'method', 'ppo_plus_eco',
    ['ppo', 'ppo_plus_ec', 'ppo_plus_eco', 'ppo_plus_grid_oracle'],
    'Solving method to use. For DMLab scenarios, this corresponds to rows in '
    'table 1 of https://arxiv.org/pdf/1810.02274.pdf. For the Mujoco Ant '
    'scenarios, this corresponds to the columns of table S1 (only "ppo" and '
    '"ppo_plus_ec" are valid choices for ant scenarios).')

DMLAB_SCENARIOS = ['noreward', 'norewardnofire', 'sparse', 'verysparse',
                   'sparseplusdoors', 'dense1', 'dense2']
MUJOCO_ANT_SCENARIOS = ['ant_no_reward']

flags.DEFINE_enum('scenario', 'verysparse',
                  DMLAB_SCENARIOS + MUJOCO_ANT_SCENARIOS,
                  'Scenario to launch.')

flags.DEFINE_integer('run_number',
                     '1',
                     'Run number to execute.')

flags.DEFINE_integer('num_timesteps', 20000000,
                     'Number of training timesteps to run.')

flags.DEFINE_integer('num_env', 12,
                     'Number of envs to run in parallel for training the '
                     'policy.')

flags.DEFINE_string('r_networks_path',
                    None,
                    'Only meaningful for the "ppo_plus_ec" method. Path to the '
                    'root dir for pre-trained r networks. If specified, '
                    'we train the policy using those pre-trained r networks. '
                    'If not specified, we first generate the R network '
                    'training data, train the R network and then train the '
                    'policy.')

PYTHON_BINARY = 'python'


def logged_check_call(command):
  """Logs the command and calls it."""
  print('=' * 70 + '\nLaunching:\n', ' '.join(command))
  subprocess.check_call(command)


def flatten_list(to_flatten):
  # pylint: disable=g-complex-comprehension
  return [item for sublist in to_flatten for item in sublist]


def quote_gin_value(v):
  if isinstance(v, six.string_types):
    return '"{}"'.format(v)
  return v


def assemble_command(base_command, params):
  """Builds a command line to launch training.

  Args:
    base_command: list(str), command prefix.
    params: dict of param -> value. Parameters prefixed by '_gin.' are
      considered gin parameters.

  Returns:
    List of strings, the components of the command line to run.
  """
  gin_params = {param_name: param_value
                for param_name, param_value in params.items()
                if param_name.startswith('_gin.')}
  params = {param_name: param_value
            for param_name, param_value in params.items()
            if not param_name.startswith('_gin.')}
  return (base_command +
          ['--{}={}'.format(param, v)
           for param, v in params.items()] +
          flatten_list([['--gin_bindings',
                         '{}={}'.format(gin_param[len('_gin.'):],
                                        quote_gin_value(v))]
                        for gin_param, v in gin_params.items()]))


def get_ppo_params(scenario):
  """Returns the param for the 'ppo' method."""
  if scenario == 'ant_no_reward':
    return {
        'policy_architecture': 'mlp',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_single_parkour_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 5,
        '_gin.OracleExplorationReward.cell_reward_normalizer': 25,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'none',
        '_gin.train.ent_coef': 8e-6,
        '_gin.train.learning_rate': 3e-4,
        '_gin.train.nsteps': 256,
        '_gin.train.nminibatches': 4,
        '_gin.train.noptepochs': 10,
        '_gin.AntWrapper.texture_mode': 'random_tiled',
    }

  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.train.ent_coef': 0.0010941138105771857,
        '_gin.train.learning_rate': 0.00019306977288832496,
    }
  else:
    return {
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.0,
        '_gin.train.ent_coef': 0.0010941138105771857,
        '_gin.train.learning_rate': 0.00019306977288832496,
    }


def get_ppo_plus_eco_params(scenario):
  """Returns the param for the 'ppo_plus_eco' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+ECO method')

  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        'r_checkpoint': '',
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
        '_gin.create_environments.online_r_training': True,
        '_gin.RNetworkTrainer.observation_history_size': 60000,
        '_gin.RNetworkTrainer.training_interval': -1,
        '_gin.CuriosityEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RNetworkTrainer.num_epochs': 10,
    }
  else:
    return {
        'action_set': '',
        'r_checkpoint': '',
        '_gin.EpisodicMemory.capacity': 200,
        '_gin.similarity_to_memory.similarity_aggregation': 'percentile',
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
        '_gin.create_environments.online_r_training': True,
        '_gin.RNetworkTrainer.observation_history_size': 60000,
        '_gin.RNetworkTrainer.training_interval': -1,
        '_gin.CuriosityEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RNetworkTrainer.num_epochs': 10,
    }


def get_ppo_plus_grid_oracle_params(scenario):
  """Returns the param for the 'ppo_plus_grid_oracle' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+grid oracle method')
  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.05246913580246913,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.train.ent_coef': 0.0066116902624148155,
    }
  else:
    return {
        'action_set': '',
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.05246913580246913,
        '_gin.train.ent_coef': 0.0066116902624148155,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
    }


def get_generate_r_training_data_commands(scenario, r_data_workdir):
  """Generates the command lines to generate R training data."""
  level = constants.Const.find_level_by_scenario(scenario)
  cmds = []
  for split in ('R_TRAINING', 'VALIDATION'):
    params = {
        'fully_qualified_level': level.fully_qualified_name,
        'split': split,
        'action_set': 'nofire' if scenario == 'norewardnofire' else '',
        'workdir': os.path.join(r_data_workdir, split),
    }
    cmds.append(assemble_command(
        [PYTHON_BINARY,
         '-m',
         'episodic_curiosity.generate_r_training_data'],
        params))
  return cmds


def get_train_r_command(r_data_workdir, r_net_workdir):
  """Returns the command to train the R-network.

  Args:
    r_data_workdir: str, input directory containing the training data.
    r_net_workdir: str, output directory where R-network checkpoints (h5 files)
        will be stored.
  """
  params = {
      'training_data_glob': os.path.join(r_data_workdir,
                                         '*/r_training_data*.tfrecords'),
      'workdir': r_net_workdir,
      # training_data_glob will match both training and validation files. There
      # is an equal number of each, and they are sorted in train_r, so using 50%
      # of the files for validation will use one of the two datasets for
      # training and the other for validation.
      'percent_validation_files': 50,
  }
  return assemble_command(
      [PYTHON_BINARY,
       '-m',
       'episodic_curiosity.train_r'],
      params)


def get_trained_r_net_path(scenario, r_net_workdir):
  """Returns the path to the h5 file of the R network."""
  if FLAGS.r_networks_path:
    # Use a pre-trained R-network.
    assert r_net_workdir is None
    if scenario in MUJOCO_ANT_SCENARIOS:
      r_network = 'mujoco_ant/r_network_weights.01980.h5'
    else:
      assert scenario in DMLAB_SCENARIOS
      level = constants.Const.find_level_by_scenario(scenario)
      if 'explore_obstructed_goals_large' in level.fully_qualified_name:
        r_network = 'explore_obstructed_goals_large/r_network_weights.01950.h5'
      elif 'rooms_keys_doors_puzzle' in level.fully_qualified_name:
        r_network = 'rooms_keys_doors_puzzle/r_network_weights.01350.h5'
      elif 'rooms_collect_good_objects_train' in level.fully_qualified_name:
        r_network = ('rooms_collect_good_objects_train/'
                     'r_network_weights.02010.h5')
      else:
        r_network = 'explore_goal_locations_large/r_network_weights.01860.h5'
    return os.path.join(FLAGS.r_networks_path, r_network)

  assert r_net_workdir is not None
  # Return the latest R-net in r_net_workdir.
  files = [os.path.join(r_net_workdir, f)
           for f in os.listdir(r_net_workdir)
           if re.search(r'r_network_weights\.\d{5}\.h5', f)]
  if not files:
    raise ValueError('No R-net found in {}'.format(r_net_workdir))
  # Take the last checkpoint.
  return max(files)


def get_ppo_plus_ec_params(scenario, r_network_path):
  """Returns the param for the 'ppo_plus_ec' method."""
  if scenario == 'ant_no_reward':
    return {
        'policy_architecture': 'mlp',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_single_parkour_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 5,
        '_gin.OracleExplorationReward.cell_reward_normalizer': 25,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'episodic_curiosity',
        '_gin.EpisodicMemory.capacity': 1000,
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.similarity_to_memory.similarity_aggregation': 'nth_largest',
        '_gin.CuriosityEnvWrapper.similarity_threshold': 1.0,
        '_gin.train.nsteps': 256,
        '_gin.train.nminibatches': 4,
        '_gin.train.noptepochs': 10,
        '_gin.CuriosityEnvWrapper.bonus_reward_additive_term': 0.5,
        'r_checkpoint': r_network_path,
        '_gin.AntWrapper.texture_mode': 'random_tiled',
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 1.0,
        '_gin.train.ent_coef': 2.23872113857e-05,
        '_gin.train.learning_rate': 7.49894209332e-05,
    }

  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'r_checkpoint': r_network_path,
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
    }
  else:
    return {
        'r_checkpoint': r_network_path,
        'action_set': '',
        '_gin.EpisodicMemory.capacity': 200,
        '_gin.similarity_to_memory.similarity_aggregation': 'percentile',
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward':
            0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
    }


def run_r_net_training(workdir):
  """Launches R-net data generation and R-net training.

  Args:
    workdir: Directory where R training data and snapshotted R networks will be
      written.

  Returns:
    Path to the trained R-networks.
  """
  # We need to train the r-networks:
  r_data_workdir = os.path.join(workdir, 'r_training_data')
  r_net_workdir = os.path.join(workdir, 'r_networks')
  for cmd in get_generate_r_training_data_commands(
      FLAGS.scenario,
      r_data_workdir):
    logged_check_call(cmd)
  logged_check_call(get_train_r_command(r_data_workdir, r_net_workdir))
  return r_net_workdir


def run_training():
  """Runs training accordding to flags."""
  workdir = os.path.join(os.path.expanduser(FLAGS.workdir),
                         FLAGS.method,
                         FLAGS.scenario,
                         'run_number_{}'.format(FLAGS.run_number))

  r_net_workdir = None
  if FLAGS.method == 'ppo_plus_ec' and not FLAGS.r_networks_path:
    assert FLAGS.scenario in DMLAB_SCENARIOS, (
        'As of today, the code does not support R-network training for '
        'non-DMLab scenarios. You can use provided checkpoints instead.')
    r_net_workdir = run_r_net_training(workdir)

  if FLAGS.method == 'ppo_plus_eco':
    policy_training_params = get_ppo_plus_eco_params(FLAGS.scenario)
  elif FLAGS.method == 'ppo':
    policy_training_params = get_ppo_params(FLAGS.scenario)
  elif FLAGS.method == 'ppo_plus_grid_oracle':
    policy_training_params = get_ppo_plus_grid_oracle_params(FLAGS.scenario)
  elif FLAGS.method == 'ppo_plus_ec':
    policy_training_params = get_ppo_plus_ec_params(
        FLAGS.scenario,
        get_trained_r_net_path(FLAGS.scenario, r_net_workdir))
  else:
    raise NotImplementedError(
        'method {} is not implemented.'.format(FLAGS.method))

  if FLAGS.scenario in DMLAB_SCENARIOS:
    env_name = ('dmlab:' + constants.Const.find_level_by_scenario(
        FLAGS.scenario).fully_qualified_name)
  else:
    assert FLAGS.scenario in MUJOCO_ANT_SCENARIOS, FLAGS.scenario
    env_name = 'parkour:'

  policy_training_params.update({
      'workdir': workdir,
      'num_env': str(FLAGS.num_env),
      'env_name': env_name,
      'num_timesteps': str(FLAGS.num_timesteps)})
  print('Params for scenario', FLAGS.scenario, ':\n', policy_training_params)
  tf.gfile.MakeDirs(workdir)
  base_command = [PYTHON_BINARY, '-m', 'episodic_curiosity.train_policy']
  logged_check_call(assemble_command(
      base_command, policy_training_params))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected command line arguments.')

  if not tf.gfile.Exists(FLAGS.workdir):
    tf.gfile.MakeDirs(FLAGS.workdir)

  run_training()


if __name__ == '__main__':
  app.run(main)
