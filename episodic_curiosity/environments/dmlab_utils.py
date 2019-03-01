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

"""Helper functions to facilitate running DMLab env.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tempfile
from absl import flags
from episodic_curiosity import oracle
from third_party.baselines import logger
from third_party.baselines.bench import Monitor
from third_party.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import numpy as np
import gin.tf
import deepmind_lab
import cv2

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'renderer', 'hardware', ['software', 'hardware'],
    'DMLab renderer. Make sure you have GPU if you use '
    '"hardware".')


def create_env_settings(level_name, homepath='', width=96, height=72, seed=0,
                        main_observation='RGB_INTERLEAVED'):
  """Creates environment settings."""
  env_settings = {
      'seed':
          seed,
      # See available levels:
      # https://github.com/deepmind/lab/tree/master/game_scripts/levels
      'levelName':
          level_name,
      'width':
          width,
      'height':
          height,
      # Documentation about the available observations:
      # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
      'observationFormat': [
          main_observation,
          'DEBUG.POS.TRANS',
          'MAP_FRAME_NUMBER',
          'DEBUG.MAZE.LAYOUT',
          'DEBUG.POS.ROT',
          'DEBUG.PLAYERS.VELOCITY',
      ],
      'homepath':
          homepath,
      'renderer':
          FLAGS.renderer,
  }
  return env_settings


# A set of allowed actions.
DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

DEFAULT_ACTION_SET_WITH_IDLE = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire.
    (0, 0, 0, 0, 0, 0, 0),  # Idle.
)

# Default set without "Fire".
DEFAULT_ACTION_SET_WITHOUT_FIRE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
)

# A small action set.
ACTION_SET_SMALL = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
)


# Another set of actions with idle.
ACTION_SET_WITH_IDLE = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (0, 0, 0, 0, 0, 0, 0),    # Idle.
)


ACTION_SET_SMALL_WITH_BACK = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
)


@gin.configurable
class DMLabWrapper(gym.Env):
  """A wrapper around DMLab environment to make it compatible with OpenAI
     Baseline's training.
  """

  def __init__(self, platform, args,
               action_set=DEFAULT_ACTION_SET,
               main_observation='RGB_INTERLEAVED',
               action_repeat=4,
               noise_type='',
               tv_num_images=30):
    """Creates a DMLabWrapper.

    Args:
      platform: Typically 'dmlab'.
      args: The environment settings.
      action_set: The set of discrete actions.
      main_observation: The observation returned at every time step.
      action_repeat: Maximum number of times to repeat an action.
        This can be less at the end of an episode.
      noise_type: if not empty defines what type of noise to add to the
        observation. Possible values: image_action, image, noise_action, noise.
      tv_num_images: number of distinct images to be used for TV purposes.
    """
    homepath = args.pop('homepath')
    level_name = args.pop('levelName')
    observation_format = args.pop('observationFormat')
    renderer = args.pop('renderer')
    seed = args.pop('seed')
    string_args = {key: str(value) for key, value in args.items()}
    if homepath:
      deepmind_lab.set_runfiles_path(os.path.join(
          homepath,
      ))
    self._env = deepmind_lab.Lab(
        level_name, observation_format, string_args, renderer)

    self._random_state = np.random.RandomState(seed=seed)
    self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))

    self._action_set = action_set
    self._action_repeat = action_repeat
    self.width = args['width']
    self.height = args['height']

    self._main_observation = main_observation
    self._transform_observation = lambda x: x
    if main_observation == 'DEBUG.CAMERA.PLAYER_VIEW_NO_RETICLE':
      # This observation format is (RGB, height, width).
      # Convert it to (height, width, RGB).
      self._transform_observation = lambda x: np.moveaxis(x, 0, -1)

    # Build a list of all the possible actions.
    self._action_list = []
    for action in action_set:
      self._action_list.append(np.array(action, dtype=np.intc))

    self._noise_type = noise_type
    self._images_for_noise = []
    if self._noise_type:
      if 'action' in self._noise_type:
        assert action_set in [DEFAULT_ACTION_SET_WITH_IDLE, DEFAULT_ACTION_SET]
      for image in range(1, tv_num_images + 1):
        image_path = '/cns/vz-d/home/raveman/images/%d.jpeg' % image
        tmp_path = os.path.join(tempfile.gettempdir(),
                                os.path.basename(image_path))
        image = cv2.imread(tmp_path, flags=cv2.IMREAD_COLOR)
        image = cv2.resize(image, (int(self.width/2), int(self.height/2)),
                           interpolation=cv2.INTER_AREA)
        # imread returns BGR not RGB
        image = image[Ellipsis, ::-1]
        self._images_for_noise.append(image)

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(self._action_set))

  @property
  def observation_space(self):
    return gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3),
                          dtype=np.uint8)

  def add_noise(self, observation, action):
    # Last action is used as "switch TV" action. Ideally it should be defined as
    # "idle", so it would not do anything else.
    if (action + 1 == len(self._action_list) or
        'action' not in self._noise_type):

      if 'image' in self._noise_type:
        self._current_noise = self._images_for_noise[
            self._random_state.randint(0, len(self._images_for_noise))]

      if 'noise' in self._noise_type:
        self._current_noise = self._random_state.randint(
            0, 255, (int(observation.shape[0]/2), int(observation.shape[1]/2),
                     observation.shape[2]))

    observation[int(observation.shape[0]/2):,
                int(observation.shape[1]/2):, :] = self._current_noise
    return observation

  def reset(self):
    self._current_noise = np.zeros((int(self.observation_space.shape[0]/2),
                                    int(self.observation_space.shape[1]/2),
                                    self.observation_space.shape[2]))
    self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
    time_step = self._env.observations()
    main_observation = self._transform_observation(
        time_step[self._main_observation])
    if self._noise_type:
      main_observation = self.add_noise(main_observation, -1)
    return main_observation

  def close(self):
    self._env.close()

  def step(self, action):
    """Performs one step in the environment.

    Args:
      action: which action to take
    Returns:
      A tuple (observation, reward, done, metadata)
    """
    reward = self._env.step(self._action_list[action],
                            num_steps=self._action_repeat)
    done = np.array(not self._env.is_running())
    if done:
      self.reset()
    time_step = self._env.observations()
    main_observation = self._transform_observation(
        time_step[self._main_observation])
    if self._noise_type:
      main_observation = self.add_noise(main_observation, action)
    metadata = {
        'position': time_step['DEBUG.POS.TRANS'],
        'frame_num': time_step['MAP_FRAME_NUMBER'],
        'maze_layout': time_step['DEBUG.MAZE.LAYOUT'],
        'rotation': time_step['DEBUG.POS.ROT'],
        'velocity': time_step['DEBUG.PLAYERS.VELOCITY'],
    }
    return (main_observation, reward, done, metadata)


class OracleRewardWrapper(gym.Wrapper):
  """Replaces reward in the environment with reward for visiting new states."""

  def __init__(self, env):
    """Creates a new oracle to compute the exploration reward."""
    gym.Wrapper.__init__(self, env)
    self._oracle_exploration_reward = oracle.OracleExplorationReward()

  def reset(self):
    self._oracle_exploration_reward.reset()
    return self.env.reset()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation, self.reward(reward, info['position']), done, info

  def reward(self, reward, agent_position):
    return self._oracle_exploration_reward.update_position(agent_position)


class EndEpisodeOnRespawn(gym.Wrapper):
  """Wrappers that end episodes on respawn."""

  def __init__(self, env):
    """Creates a new wrapper that terminates episodes on respawn."""
    gym.Wrapper.__init__(self, env)
    self._last_frame_num = 0

  def reset(self):
    self._last_frame_num = 0
    return self.env.reset()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)

    frame_num = info['frame_num']
    if done:
      # Set the last frame num to 0 so that the next step is not
      # considered as a respawn.
      self._last_frame_num = 0
    else:
      if frame_num < self._last_frame_num:
        # A frame number that is decreasing means there was a respawn.
        # Set the done flag and resets the underlying environment.
        print('Respawn detected fn: ', frame_num, ' lfn: ',
              self._last_frame_num, ' reward:', reward)
        self.reset()
        done = True
      else:
        self._last_frame_num = frame_num

    return observation, reward, done, info


def make_dmlab_env(env_settings, num_env=8, small_action_set=False,
                   oracle_reward=False, use_monitor=True):
  """Creates a DMLab environment."""
  def make_env(seed):
    def _thunk():
      tmp_settings = copy.deepcopy(env_settings)
      tmp_settings['seed'] = seed
      action_set = ACTION_SET_SMALL if small_action_set else DEFAULT_ACTION_SET
      env = DMLabWrapper('dmlab', tmp_settings, action_set=action_set)
      if oracle_reward:
        env = OracleRewardWrapper(env)
      if use_monitor:
        env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                             str(seed)))
      return env
    return _thunk

  return SubprocVecEnv([make_env(1 + i) for i in range(num_env)])
