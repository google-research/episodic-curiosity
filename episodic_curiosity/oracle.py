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

"""Computes some oracle reward based on the actual agent position."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin


@gin.configurable
class OracleExplorationReward(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self, reward_grid_size=30.0):
    """Creates a new oracle to compute the exploration reward.

    Args:
      reward_grid_size: Size of a cell that contains a unique reward.
    """
    self._reward_grid_size = reward_grid_size

    # Make the total sum of exploration reward that can be collected
    # independent of the grid size.
    # Here, we assume that the position is laying on a 2D manifold,
    # hence the multiplication by the area of a 2D cell.
    self._cell_reward = float(reward_grid_size * reward_grid_size)

    # Somewhat normalize the exploration reward so that it is neither
    # too big or too small.
    # Note: this is DMLab specific. This gives a reward of 1.0
    # per cell when the grid_size is set to 30.
    self._cell_reward /= (30.0 * 30.0)

    self.reset()

  def reset(self):
    self._collected_positions = set()

  def update_position(self, agent_position):
    """Set the new state (i.e. the position).

    Args:
      agent_position: x,y,z position of the agent.

    Returns:
      The exploration bonus for having visited this position.
    """
    x, y, z = agent_position
    quantized_x = int(x / self._reward_grid_size)
    quantized_y = int(y / self._reward_grid_size)
    quantized_z = int(z / self._reward_grid_size)
    position_id = (quantized_x, quantized_y, quantized_z)
    if position_id in self._collected_positions:
      # No reward if the position has already been explored.
      return 0.0
    else:
      self._collected_positions.add(position_id)
      return self._cell_reward
