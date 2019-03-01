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

"""Tests for dune.rl.episodic_curiosity.constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from episodic_curiosity import constants


class ConstantsTest(absltest.TestCase):

  def test_unique_levels(self):
    unique_levels = set()
    for level in constants.Const.LEVELS:
      self.assertNotIn(level.fully_qualified_name, unique_levels)
      unique_levels.add(level.fully_qualified_name)

  def test_find_level(self):
    self.assertEqual(
        constants.Const.find_level('explore_goal_locations_large')
        .dmlab_level_name, 'contributed/dmlab30/explore_goal_locations_large')


if __name__ == '__main__':
  absltest.main()
