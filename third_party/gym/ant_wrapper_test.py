# coding=utf-8
# The MIT License
#
# Copyright (c) 2016 OpenAI (https://openai.com)
# Copyright (c) 2018 The TF-Agents Authors.
# Copyright (c) 2018 Google LLC (http://google.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWIS, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Tests for google3.third_party.py.third_party.gym.ant_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from third_party.gym import ant_wrapper
from google3.pyglib import resources
from google3.testing.pybase import googletest

ASSETS_DIR = 'google3/third_party/py/third_party.gym/assets'


def get_resource(filename):
  return resources.GetResourceFilenameInDirectoryTree(
      os.path.join(ASSETS_DIR, filename))


class AntWrapperTest(googletest.TestCase):

  def test_ant_wrapper(self):
    env = ant_wrapper.AntWrapper(
        get_resource('mujoco_ant_custom_texture_camerav2.xml'),
        texture_mode='fixed',
        texture_file_pattern=get_resource('texture.png'))
    env.reset()
    obs, unused_reward, unused_done, info = env.step(env.action_space.sample())
    self.assertEqual(obs.shape, (27,))
    self.assertIn('frame', info)
    self.assertEqual(info['frame'].shape,
                     (120, 160, 3))


if __name__ == '__main__':
  googletest.main()
