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

"""Checks that DMLab levels specified in constants are valid."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from episodic_curiosity import constants
from episodic_curiosity.environments import dmlab_utils


class ConstantsTest(absltest.TestCase):

  def test_levels_exist(self):
    for level in constants.Const.LEVELS:
      self._check_level_exists(level)

  def _check_level_exists(self, level):
    settings = dmlab_utils.create_env_settings(level.dmlab_level_name, seed=1)
    settings.update(level.extra_env_settings)
    env = dmlab_utils.DMLabWrapper('dmlab', settings)
    env.reset()


if __name__ == '__main__':
  absltest.main()
