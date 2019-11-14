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

"""Ant wrappers for episodic curiosity.

This environment is described in section S1 of
https://arxiv.org/abs/1810.02274.pdf.

Compared to the common ant environments, this adds a head camera, and multiple
ways to tile the floor.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import random
import tempfile
from absl import flags
from dm_control import mujoco
from third_party.gym import ant
try:
  # Python 2
  # pylint: disable=g-import-not-at-top
  import functools32 as functools
except ImportError:
  # Python 3
  # pylint: disable=g-import-not-at-top
  import functools  # type: ignore
# pylint: disable=g-import-not-at-top
import gin
import gym
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('asset_path', '',
                    'Path to assets (images, walker definitions).')


# Number of actions for the ant mujoco model.
_NUM_ANT_ACTIONS = 8
# Number of observation points for the ant mujoco model.
_ANT_OBS_DIM = 27
# Min/max ids of textures.
MIN_TEXTURE_ID = 1
MAX_TEXTURE_ID = 190

# Size of each tile (in pixels). Used when texture_mode='random_tiled' is used.
TILED_IMG_SIZE = 100
# Floor size in environment units. Used when random_start_position=True.
FLOOR_SIZE_IN_ENV = 200




# We cache the decoded images in memory, so that we don't have to re-read and
# resize tens of large images at each episode. This is a module function so that
# the cache is shared, for efficiency.
@functools.lru_cache(maxsize=1024)
def _read_resized_texture_content(texture_path):
  """Reads and resizes a texture."""
  original = skimage.io.imread(texture_path)
  resized = skimage.transform.resize(
      original, (TILED_IMG_SIZE, TILED_IMG_SIZE), mode='constant')
  if len(resized.shape) == 2:
    # Grayscale image, convert to RGB.
    resized = skimage.color.gray2rgb(resized)
  return resized


@gin.configurable
class AntWrapper(gym.Env):
  """Gym API for a customized Mujoco ant."""

  def __init__(self,
               walker_basename = 'mujoco_ant_custom_texture_camerav2.xml',
               camera_names = None,
               texture_mode = 'random_tiled',
               texture_file_pattern = 't_{}.png',
               texture_size_in_tiles = 20,
               random_start_position = True,
               height = 120,
               width = 160,
               mujoco_key_path = None):
    """Creates an ant wrapper.

    Args:
      walker_basename: Basename of the file (in the asset path) containing the
        xml definition for the walker.
      camera_names: Names of the cameras used to generate observations. These
        cameras must be defined in the walker definition.
      texture_mode: One of the following:
        - 'random': a new texture will be used for the floor at each episode.
        - 'fixed': we always use the same texture for the floor.
        - 'random_tiled': N*N textures will be randomly selected and
          tiled, with N=texture_size_in_tiles.
      texture_file_pattern: Basename for the texture. When texture_mode !=
        'fixed', the string will be formatted to include an ID in
        [MIN_TEXTURE_ID, MAX_TEXTURE_ID]. We expect to find them under
        FLAGS.parkour_assets/textures/.
      texture_size_in_tiles: Number of horizontal and vertical tiles that are
        assembled to form the floor texture. Only used when
        texture_mode='random_tiled'.
      random_start_position: If true, the starting position of the ant will be
        chosen randomly.
      height: Height (pixels) of the observations.
      width: Width (pixels) of the observations.
      mujoco_key_path: Optional path to license file. If not provided,
        dm_control.mujoco will automatically use the key in the resources of the
        python binary. This can fail if this environment is launched in parallel
        from multiple forked processes. This is why you can provide an explicit
        path to the license to fix this issue.
    """
    if camera_names is None:
      camera_names = ['head_camera', 'track']
    self._camera_names = camera_names
    self.height = height
    self.width = width
    if texture_mode == 'fixed':
      self._texture_paths = [self._full_texture_path(texture_file_pattern)]
    elif texture_mode == 'random' or texture_mode == 'random_tiled':
      self._texture_paths = [
          self._full_texture_path(texture_file_pattern.format(i))
          for i in range(MIN_TEXTURE_ID, MAX_TEXTURE_ID)
      ]
    else:
      raise ValueError('Unsupported texture_mode: {}'.format(texture_mode))
    self._texture_mode = texture_mode
    self._texture_size_in_tiles = texture_size_in_tiles
    self._random_start_position = random_start_position
    with tf.gfile.Open(
        os.path.join(FLAGS.asset_path, walker_basename)) as f:
      self._walker_model_template = f.read()
    if mujoco_key_path:
      os.environ['MJKEY_PATH'] = mujoco_key_path
    self._cur_env_step = 0
    # Tmp paths to clean up before each new episode.
    self._tmp_episode_paths = []
    self._init_env()

  def _full_texture_path(self, texture_basename):
    return os.path.join(FLAGS.asset_path,
                        'textures', texture_basename)

  def _get_texture(self):
    """Returns (path,should_delete) for the texture to use for a new episode."""
    if self._texture_mode != 'random_tiled':
      return (random.choice(self._texture_paths), False)

    # We are in 'random_tiled' mode: we create an image by tiling a random
    # subset of textures.
    tiled_texture = np.zeros(shape=(
        self._texture_size_in_tiles * TILED_IMG_SIZE,
        self._texture_size_in_tiles * TILED_IMG_SIZE,
        3),
                             dtype=np.float32)
    for i in range(self._texture_size_in_tiles):
      for j in range(self._texture_size_in_tiles):
        texture_path = random.choice(self._texture_paths)
        texture_contents = _read_resized_texture_content(texture_path)
        tiled_texture[i * TILED_IMG_SIZE: (i + 1) * TILED_IMG_SIZE,
                      j * TILED_IMG_SIZE: (j + 1) * TILED_IMG_SIZE,
                      :] = texture_contents
    with tempfile.NamedTemporaryFile(
        prefix='tiled_texture_', suffix='.png',
        delete=False) as tmp_file:
      tiled_texture_path = tmp_file.name
    skimage.io.imsave(tiled_texture_path, tiled_texture)
    return (tiled_texture_path, True)

  def _gen_start_coord(self):
    """Generates a starting coordinate for the ant."""
    if self._random_start_position:
      return random.uniform(-1, 1) * FLOOR_SIZE_IN_ENV / 2
    else:
      return 0.

  def _init_env(self):
    """Initializes a new environment with a randomly chosen texture."""
    for path in self._tmp_episode_paths:
      # Cleanup previous tmp paths, so that they don't accumulate. We don't
      # clean them up directly after instanciating the Ant, before it's not
      # clear when and how often mujoco's code will read the files it's given.
      tf.gfile.Remove(path)
    del self._tmp_episode_paths[:]

    texture_path, should_delete = self._get_texture()
    if should_delete:
      self._tmp_episode_paths.append(texture_path)
    walker_model = self._walker_model_template.format(
        texture_file=texture_path,
        start_x=self._gen_start_coord(),
        start_y=self._gen_start_coord())
    with tempfile.NamedTemporaryFile(
        prefix='ant_', suffix='.xml', delete=False) as tmp_file:
      tmp_model_path = tmp_file.name
      print('Writing temporary walker definition:', tmp_model_path)
      tmp_file.write(walker_model.encode())

    self._tmp_episode_paths.append(tmp_model_path)
    self._ant = AntWithCustomCamera(
        model_path=tmp_model_path,
        height=self.height,
        width=self.width,
        camera_ids=self._camera_names)

  @property
  def action_space(self):
    return gym.spaces.Box(-1, 1, shape=(_NUM_ANT_ACTIONS,), dtype=np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(-1, 1, shape=(_ANT_OBS_DIM,), dtype=np.float32)

  def reset(self):
    """Resets the environment (see gym API for details)."""
    # We need to reload the env to get a a random texture at each episode, or
    # random initial position (if need be).
    self._init_env()
    ob = self._ant.reset()
    # Obs starting from _ANT_OBS_DIM are external forces applied to the ant.
    # Since we don't apply external forces, they are all 0s.
    ob = ob[:_ANT_OBS_DIM]
    self._cur_env_step = 0
    return ob

  def step(self, action):
    """Step the environment (see gym API for details)."""
    ob, reward, done, info = self._ant.step(action)
    for i, camera in enumerate(self._camera_names):
      frame = self._ant.render('rgb_array', camera)
      info['frame:' + camera] = frame
      if i == 0:
        info['frame'] = frame
    info['position'] = self._ant.get_body_com('torso')
    # MujocoEnv has max_episode_steps=1000 (set from AntEnv), but does not seem
    # to use it...
    if self._cur_env_step >= self._ant.max_episode_steps:  # type: ignore
      done = True
    self._cur_env_step += 1
    return ob[:_ANT_OBS_DIM], reward, done, info


class AntWithCustomCamera(ant.AntEnv):
  """AntEnv with custom camera options."""

  def __init__(self,
               model_path='ant.xml',
               height=120,
               width=160,
               camera_ids=None):
    if camera_ids is None:
      camera_ids = ['head_camera']
    self._camera_ids = camera_ids
    if not self._camera_ids:
      raise ValueError('We need at least one camera ID')
    self._height = height
    self._width = width
    ant.AntEnv.__init__(self, model_path=model_path)

    self.all_cameras = {
        camera_id: mujoco.Camera(
            self.physics,
            height=self._height,
            width=self._width,
            camera_id=camera_id) for camera_id in self._camera_ids
    }

  def render(self, mode='human', camera_id=None):
    if mode != 'rgb_array':
      return ant.AntEnv.render(mode)
    if camera_id is None:
      camera_id = self._camera_ids[0]
    data = self.all_cameras[camera_id].render()  # type: ignore
    return np.copy(data)

  def camera_setup(self):
    self.camera = mujoco.Camera(
        self.physics,
        height=self._height,
        width=self._width,
        camera_id=self._camera_ids[0])
