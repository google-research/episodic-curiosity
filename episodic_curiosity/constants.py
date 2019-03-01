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

"""Constants for episodic curiosity."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from enum import Enum


class Level(object):
  """Represents a DMLab level, possibly with additional non-standard settings.

  Attributes:
    dmlab_level_name: Name of the DMLab level
    fully_qualified_name: Unique name used to distinguish between multiple DMLab
                          levels with the same name but different settings.
    extra_env_settings: dict, additional DMLab environment settings for this
                        level.
    random_maze: Whether the geometry of the maze is supposed to change when we
                 change the seed.
    use_r_net_from_level: If provided, don't train a R-net for this level, but
                          instead, use the trained R-net from another level
                          (identified by its fully qualified name).
    include_in_paper: Whether this level is included in the paper.
    scenarios: Optional list of scenarios this level is used for.
  """

  def __init__(self,
               dmlab_level_name,
               fully_qualified_name = None,
               extra_env_settings = None,
               random_maze = False,
               use_r_net_from_level = None,
               include_in_paper = False,
               scenarios = None):
    self.dmlab_level_name = dmlab_level_name
    self.fully_qualified_name = fully_qualified_name or dmlab_level_name
    self.extra_env_settings = extra_env_settings or {}
    self.random_maze = random_maze
    self.use_r_net_from_level = use_r_net_from_level
    self.include_in_paper = include_in_paper
    self.scenarios = scenarios

  def asdict(self):
    return vars(self)


class SplitType(Enum):
  R_TRAINING = 0
  POLICY_TRAINING = 3
  VALIDATION = 1
  TEST = 2


class Const(object):
  """Constants"""
  MAX_ACTION_DISTANCE = 5
  NEGATIVE_SAMPLE_MULTIPLIER = 5
  # env
  OBSERVATION_HEIGHT = 120
  OBSERVATION_WIDTH = 160
  OBSERVATION_CHANNELS = 3
  OBSERVATION_SHAPE = (OBSERVATION_HEIGHT, OBSERVATION_WIDTH,
                       OBSERVATION_CHANNELS)
  # model and training
  BATCH_SIZE = 64
  EDGE_CLASSES = 2
  DUMP_AFTER_BATCHES = 100
  EDGE_MAX_EPOCHS = 2000
  ADAM_PARAMS = {
      'lr': 1e-04,
      'beta_1': 0.9,
      'beta_2': 0.999,
      'epsilon': 1e-08,
      'decay': 0.0
  }
  ACTION_REPEAT = 4
  STORE_CHECKPOINT_EVERY_N_EPOCHS = 30

  LEVELS = [
      # Levels on which we evaluate episodic curiosity.
      # Corresponds to 'Sparse' setting in the paper
      # (arxiv.org/pdf/1810.02274.pdf).
      Level('contributed/dmlab30/explore_goal_locations_large',
            fully_qualified_name='explore_goal_locations_large',
            random_maze=True,
            include_in_paper=True,
            scenarios=['sparse', 'noreward', 'norewardnofire']),

      # WARNING!! For explore_goal_locations_large_sparse and
      # explore_goal_locations_large_verysparse to work properly (i.e. taking
      # into account minGoalDistance), you need to use the dmlab MPM:
      # learning/brain/research/dune/rl/dmlab_env_package.
      # Corresponds to 'Very Sparse' setting in the paper.
      Level(
          'contributed/dmlab30/explore_goal_locations_large',
          fully_qualified_name='explore_goal_locations_large_verysparse',
          extra_env_settings={
              # Forces the spawn and goals to be further apart.
              # Unfortunately, we cannot go much higher, because we need to
              # guarantee that for any goal location, we can at least find one
              # spawn location that is further than this number (the goal
              # location might be in the middle of the map...).
              'minGoalDistance': 10,
          },
          use_r_net_from_level='explore_goal_locations_large',
          random_maze=True, include_in_paper=True,
          scenarios=['verysparse']),

      # Corresponds to 'Sparse+Doors' setting in the paper.
      Level('contributed/dmlab30/explore_obstructed_goals_large',
            fully_qualified_name='explore_obstructed_goals_large',
            random_maze=True,
            include_in_paper=True,
            scenarios=['sparseplusdoors']),

      # Two levels where we expect to show episodic curiosity does not hurt.
      # Corresponds to 'Dense 1' setting in the paper.
      Level('contributed/dmlab30/rooms_keys_doors_puzzle',
            fully_qualified_name='rooms_keys_doors_puzzle',
            include_in_paper=True,
            scenarios=['dense1']),
      # Corresponds to 'Dense 2' setting in the paper.
      Level('contributed/dmlab30/rooms_collect_good_objects_train',
            fully_qualified_name='rooms_collect_good_objects_train',
            include_in_paper=True,
            scenarios=['dense2']),
  ]

  MIXER_SEEDS = {
      # Equivalent to not setting a mixer seed. Mixer seed to train the
      # R-network.
      SplitType.R_TRAINING: 0,
      # Mixer seed for training the policy.
      SplitType.POLICY_TRAINING: 0x3D23BE66,
      SplitType.VALIDATION: 0x2B79ED94,  # Invented.
      SplitType.TEST: 0x600D5EED,  # Same as DM's.
  }

  @staticmethod
  def find_level(fully_qualified_name):
    """Finds a DMLab level by fully qualified name."""
    for level in Const.LEVELS:
      if level.fully_qualified_name == fully_qualified_name:
        return level
    # Fallback to the DMLab level with the corresponding name.
    return Level(fully_qualified_name,
                 extra_env_settings = {
                     # Make 'rooms_exploit_deferred_effects_test',
                     # 'rooms_collect_good_objects_test' work.
                     'allowHoldOutLevels': True
                 })

  @staticmethod
  def find_level_by_scenario(scenario):
    """Finds a DMLab level by scenario name."""
    for level in Const.LEVELS:
      if level.scenarios and scenario in level.scenarios:
        return level
    raise ValueError('Scenario "{}" not found.'.format(scenario))
