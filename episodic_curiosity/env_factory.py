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

"""Factories to create DMLab env with episodic curiosity rewards."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import functools
import os
from absl import flags
from episodic_curiosity import constants
from episodic_curiosity import curiosity_env_wrapper
from episodic_curiosity import episodic_memory
from episodic_curiosity import r_network
from episodic_curiosity import r_network_training
from episodic_curiosity.constants import Const
from episodic_curiosity.environments import dmlab_utils
from third_party.baselines import logger
from third_party.baselines.bench import Monitor
from third_party.baselines.common import atari_wrappers
from third_party.baselines.common.vec_env import subproc_vec_env
from third_party.baselines.common.vec_env import threaded_vec_env
from third_party.gym import ant_wrapper
from third_party.keras_resnet import models
import gin

DEFAULT_VEC_ENV_CLASS_NAME = 'SubprocVecEnv'


flags.DEFINE_enum('vec_env_class',
                  DEFAULT_VEC_ENV_CLASS_NAME,
                  ['SubprocVecEnv', 'ThreadedVecEnv'],
                  'Vec env class to use. ')

FLAGS = flags.FLAGS


def get_action_set(action_set_name):
  """Returns action sets by name."""
  return {
      '': dmlab_utils.DEFAULT_ACTION_SET,
      'small': dmlab_utils.ACTION_SET_SMALL,
      'nofire': dmlab_utils.DEFAULT_ACTION_SET_WITHOUT_FIRE,
      'withidle': dmlab_utils.ACTION_SET_WITH_IDLE,
      'defaultwithidle': dmlab_utils.DEFAULT_ACTION_SET_WITH_IDLE,
      'smallwithback': dmlab_utils.ACTION_SET_SMALL_WITH_BACK,
  }[action_set_name]


@gin.configurable
def create_single_env(env_name, seed, dmlab_homepath, use_monitor,
                      split='train', vizdoom_maze=False, action_set='',
                      respawn=True, fixed_maze=False, maze_size=None,
                      room_count=None, episode_length_seconds=None,
                      min_goal_distance=None, run_oracle_before_monitor=False):
  """Creates a single instance of DMLab env, with training mixer seed.

  Args:
    env_name: Name of the DMLab environment.
    seed: seed passed to DMLab. Must be != 0.
    dmlab_homepath: Path to DMLab MPM. Required when running on borg.
    use_monitor: Boolean to add a Monitor wrapper.
    split: One of {"train", "valid", "test"}.
    vizdoom_maze: Whether a geometry of a maze should correspond to the one used
      by Pathak in his curiosity paper in Vizdoom environment.
    action_set: One of {'small', 'nofire', ''}. Which action set to use.
    respawn: If disabled respawns are not allowed
    fixed_maze: Boolean to use predefined maze configuration.
    maze_size: If not None sets particular height/width for mazes to be
      generated.
    room_count: If not None sets the number of rooms for mazes to be generated.
    episode_length_seconds: If not None overrides the episode duration.
    min_goal_distance: If not None ensures that there's at least this distance
      between the starting and target location (for
      explore_goal_locations_large level).
    run_oracle_before_monitor: Whether to run OracleRewardWrapper before the
      Monitor.

  Returns:
    Gym compatible DMLab env.

  Raises:
    ValueError: when the split is invalid.
  """
  main_observation = 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE'
  level = constants.Const.find_level(env_name)
  env_settings = dmlab_utils.create_env_settings(
      level.dmlab_level_name,
      homepath=dmlab_homepath,
      width=Const.OBSERVATION_WIDTH,
      height=Const.OBSERVATION_HEIGHT,
      seed=seed,
      main_observation=main_observation)
  env_settings.update(level.extra_env_settings)

  if maze_size:
    env_settings['mazeHeight'] = maze_size
    env_settings['mazeWidth'] = maze_size
  if min_goal_distance:
    env_settings['minGoalDistance'] = min_goal_distance
  if room_count:
    env_settings['roomCount'] = room_count
  if episode_length_seconds:
    env_settings['episodeLengthSeconds'] = episode_length_seconds

  if split == 'train':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.POLICY_TRAINING]
  elif split == 'valid':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.VALIDATION]
  elif split == 'test':
    mixer_seed = Const.MIXER_SEEDS[constants.SplitType.TEST]
  else:
    raise ValueError('Invalid split: {}'.format(split))
  env_settings.update(mixerSeed=mixer_seed)

  if vizdoom_maze:
    env_settings['episodeLengthSeconds'] = 60
    env_settings['overrideEntityLayer'] = """*******************
*****   *   ***   *
*****             *
*****   *   ***   *
****** *** ***** **
*   *   *   ***   *
*P          ***   *
*   *   *   ***   *
****** ********* **
****** *********G**
*****   ***********
*****   ***********
*****   ***********
****** ************
****** ************
******   **********
*******************"""

  if fixed_maze:
    env_settings['overrideEntityLayer'] = """
*****************
*       *PPG    *
* *** * *PPP*** *
* *GPP* *GGG PGP*
* *GPG* * ***PGP*
* *PGP*   ***PGG*
* *********** * *
*     *GPG*GGP  *
* *** *PPG*PGG* *
*PGP* *GPP PPP* *
*PPP* * *** *** *
*GGG*     *GPP* *
*** ***** *GGG* *
*GPG PPG   PPP* *
*PGP*GGP* ***** *
*GPP*GPP*       *
*****************"""

  # Gym compatible environment.
  env = dmlab_utils.DMLabWrapper(
      'dmlab',
      env_settings,
      action_set=get_action_set(action_set),
      main_observation=main_observation)

  if run_oracle_before_monitor:
    env = dmlab_utils.OracleRewardWrapper(env)

  if vizdoom_maze or not respawn:
    env = dmlab_utils.EndEpisodeOnRespawn(env)

  if use_monitor:
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(seed)))
  return env


@gin.configurable
def create_single_atari_env(env_name, seed, use_monitor, split=''):
  env = atari_wrappers.make_atari(env_name)
  env.seed(seed)
  if use_monitor:
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(seed)))
  env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
  return env




@gin.configurable
def create_environments(env_name,
                        num_envs,
                        r_network_weights_path = None,
                        dmlab_homepath = '',
                        action_set = '',
                        base_seed = 123,
                        scale_task_reward_for_eval = 1.0,
                        scale_surrogate_reward_for_eval = 0.0,
                        online_r_training = False,
                        environment_engine = 'dmlab',
                        r_network_weights_store_path = ''):
  """Creates a environments with R-network-based curiosity reward.

  Args:
    env_name: Name of the DMLab environment.
    num_envs: Number of parallel environment to spawn.
    r_network_weights_path: Path to the weights of the R-network.
    dmlab_homepath: Path to the DMLab MPM. Required when running on borg.
    action_set: One of {'small', 'nofire', ''}. Which action set to use.
    base_seed: Each environment will use base_seed+env_index as seed.
    scale_task_reward_for_eval: scale of the task reward to be used for
      valid/test environments.
    scale_surrogate_reward_for_eval: scale of the surrogate reward to be used
      for valid/test environments.
    online_r_training: Whether to enable online training of the R-network.
    environment_engine: either 'dmlab', 'atari', 'parkour'.
    r_network_weights_store_path: Directory where to store R checkpoints
      generated during online training of the R network.

  Returns:
    Wrapped environment with curiosity.
  """
  # Environments without intrinsic exploration rewards.
  # pylint: disable=g-long-lambda
  create_dmlab_single_env = functools.partial(create_single_env,
                                              dmlab_homepath=dmlab_homepath,
                                              action_set=action_set)

  if environment_engine == 'dmlab':
    create_env_fn = create_dmlab_single_env
    is_atari_environment = False
  elif environment_engine == 'atari':
    create_env_fn = create_single_atari_env
    is_atari_environment = True
  else:
    raise ValueError('Unknown env engine {}'.format(environment_engine))

  # WARNING: python processes are not really compatible with other google3 code,
  # which can lead to deadlock. See go/g3process. This is why you can use
  # ThreadedVecEnv.
  VecEnvClass = (subproc_vec_env.SubprocVecEnv
                 if FLAGS.vec_env_class == 'SubprocVecEnv'
                 else threaded_vec_env.ThreadedVecEnv)

  vec_env = VecEnvClass([
      (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=True,
                                  split='train'))
      for i in range(num_envs)
  ])
  valid_env = VecEnvClass([
      (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=False,
                                  split='valid'))
      for i in range(num_envs)
  ])
  test_env = VecEnvClass([
      (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=False,
                                  split='test'))
      for i in range(num_envs)
  ])
  # pylint: enable=g-long-lambda

  # Size of states when stored in the memory.
  embedding_size = models.EMBEDDING_DIM

  if not r_network_weights_path:
    # Empty string equivalent to no R_network checkpoint.
    r_network_weights_path = None
  r_net = r_network.RNetwork(
      (84, 84, 4) if is_atari_environment else Const.OBSERVATION_SHAPE,
      r_network_weights_path)

  # Only for online training do we need to train the R-network.
  r_network_trainer = None
  if online_r_training:
    r_network_trainer = r_network_training.RNetworkTrainer(
        r_net._r_network,  # pylint: disable=protected-access
        checkpoint_dir=r_network_weights_store_path)

  # Creates the episodic memory that is attached to each of those envs.
  vec_episodic_memory = [
      episodic_memory.EpisodicMemory(
          observation_shape=[embedding_size],
          observation_compare_fn=r_net.embedding_similarity)
      for _ in range(num_envs)
  ]

  # The size of images is reduced to 64x64 to make training faster.
  # Note: using color images with DMLab makes it much easier to train a policy.
  # So no conversion to grayscale.
  target_image_shape = [84, 84, 4 if is_atari_environment else 3]
  env_wrapper = curiosity_env_wrapper.CuriosityEnvWrapper(
      vec_env, vec_episodic_memory, r_net.embed_observation, target_image_shape)
  if r_network_trainer is not None:
    env_wrapper.add_observer(r_network_trainer)

  valid_env_wrapper, test_env_wrapper = (
      curiosity_env_wrapper.CuriosityEnvWrapper(
          env, vec_episodic_memory, r_net.embed_observation,
          target_image_shape,
          exploration_reward=('none' if (is_atari_environment or
                                         environment_engine == 'parkour')
                              else 'oracle'),
          scale_task_reward=scale_task_reward_for_eval,
          scale_surrogate_reward=scale_surrogate_reward_for_eval)
      for env in [valid_env, test_env])

  return env_wrapper, valid_env_wrapper, test_env_wrapper
