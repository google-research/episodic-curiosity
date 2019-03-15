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

"""Logging for episodic curiosity."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import numpy as np


class VideoWriter(object):
  """Wrapper around video writer APIs."""

  def __init__(self, filename):
    # We don't have this library inside Google, so we only import it in the
    # Open-source version when we need it, and tell pytype to ignore the
    # fact that it's missing.
    # pylint: disable=g-import-not-at-top
    import skvideo.io  # type: ignore
    self._writer = skvideo.io.FFmpegWriter(filename)

  def add(self, frame):
    self._writer.writeFrame(frame)

  def close(self):
    self._writer.close()




def get_video_writer(video_filename):
  return VideoWriter(video_filename)  # pylint:disable=unreachable


def save_episode_buffer_as_video(episode_buffer, video_filename):
  """Saves episode_buffer."""
  video_writer = get_video_writer(video_filename)
  for frame in episode_buffer:
    video_writer.add(frame)
  video_writer.close()


def save_training_examples_as_video(training_examples, video_filename):
  """Split example into two images and show side-by-side for a while."""
  video_writer = get_video_writer(video_filename)
  for example in training_examples:
    first = example[Ellipsis, :3]
    second = example[Ellipsis, 3:]
    side_by_side = np.concatenate((first, second), axis=0)
    video_writer.add(side_by_side)
  video_writer.close()


def get_logger_dir(exp_id):
  return datetime.datetime.now().strftime('ec-%Y-%m-%d-%H-%M-%S-%f_') + exp_id
