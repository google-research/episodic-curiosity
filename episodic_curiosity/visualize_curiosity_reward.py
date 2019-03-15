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

"""Tool that builds trajectory/reward visualizations from policy checkpoints.

Example:
https://2.bp.blogspot.com/-vYTrGZe07E8/W9CinK0dkyI/AAAAAAAADcU/rRYZw30k_0IQ5SrOzamcaKdsXk4JDhutwCLcBGAs/s1600/image2.gif
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags
from absl import logging
from episodic_curiosity import curiosity_env_wrapper
from episodic_curiosity import env_factory
from episodic_curiosity import episodic_memory
from episodic_curiosity import logging as ec_logging
from episodic_curiosity import utils
from third_party.baselines.ppo2 import policies
from third_party.baselines.ppo2 import ppo2
import gin
import gym
import numpy as np
import skimage.transform
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dmlab_homepath', '',
    'Path to the DMLab resources. Only required when running '
    'on Borg.')
flags.DEFINE_string('r_net_weights', '', 'Path to the weights of the R network')
flags.DEFINE_integer('num_steps', 2000, 'Number of env steps to run.')
flags.DEFINE_string('workdir', '/tmp',
                    'Directory to which videos and trajectories are output.')
flags.DEFINE_integer('numpy_random_seed', 7, 'Seed for numpy.random.')
flags.DEFINE_integer(
    'base_env_seed', 123, 'Base DMLab seed. Each episode will use the seed '
    'base_env_seed+episode_index')
flags.DEFINE_enum('action_set', '',
                  ['', 'small', 'withidle', 'nofire', 'smallwithback'],
                  'Action set to use.')
flags.DEFINE_string('policy_path', '',
                    'Path to the checkpoint of the trained policy.')
flags.DEFINE_float(
    'eps_degraded', 0, 'Allows picking a random action with probability '
    '"eps_degraded" at each env step. 0 means we only use '
    'actions from the loaded policy. 1 means we ignore the '
    'loaded policy, and always pick random actions.')
flags.DEFINE_integer('num_episodes', 2, 'Number of episodes to run')
flags.DEFINE_enum(
    'visualization_type', 'surrogate_reward',
    ['surrogate_reward', 'observation'],
    '"surrogate_reward" creates a visualization of the surrogate '
    'rewards. It contains three parts: (1) a top-down view of a '
    'trajectory colored according to the sign of the surrogate '
    'reward, (2) a top-down view of the locations in memory '
    '(3) a first-person view with a green or red rectangle '
    'according to the sign of the surrogate reward. '
    '"observation" just shows a first-person view '
    '(vanilla environment observation)')
flags.DEFINE_enum(
    'trajectory_mode', 'save', ['do_nothing', 'save', 'load'],
    '"do_nothing": load and run the policy, but do not save the trajectory.'
    '"save": policy will be loaded and run. The sequence of '
    'actions taken by the policy will be saved to the workdir.'
    '"load": actions will be reloaded from the workdir. The '
    'actions from the loaded policy are ignored. '
    'This feature is cheap way to be able to generate '
    'high-resolution videos by replaying a trajectory generated '
    'using the resolution that the policy expects.')
flags.DEFINE_bool('use_curiosity', False,
                  'Whether to enable Pathak\'s curiosity')
flags.DEFINE_float('curiosity_strength', 0.55,
                   'Strength of the intrinsic reward in Pathak\'s algorithm.')
flags.DEFINE_float(
    'forward_inverse_ratio', 0.96,
    'Weighting of forward vs inverse loss in Pathak\'s '
    'algorithm')
flags.DEFINE_float('curiosity_loss_strength', 10,
                   'Weight of the curiosity loss in Pathak\'s algorithm.')
flags.DEFINE_string(
    'environment_engine', 'dmlab',
    'Environment engine passed to env_factory.create_environments.')
flags.DEFINE_string('policy_architecture', 'cnn',
                    'What model architecture to use')
flags.DEFINE_bool('ant_env_enable_die_condition', True,
                  'See enable_die_condition in AntEnv')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')


# Size in pixels of a single DMLab cell.
CELL_SIZE_PIXELS = 20
VIDEO_FILENAME_TEMPLATE = 'video_{}.mp4'
TRAJECTORY_FILENAME_TEMPLATE = 'trajectory_{}.pickle'
# DMLab level
LEVEL_NAME = 'explore_goal_locations_large'
# Size of the DMLab maze. Change this according to LEVEL_NAME.
LEVEL_MAZE_SIZE = 17

_GREEN = np.array([0, 255, 0], dtype=np.uint8)
_RED = np.array([255, 0, 0], dtype=np.uint8)
_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def get_obs(obs_info_pairs):
  return [obs for obs, _ in obs_info_pairs]


def visualize_curiosity_reward(work_unit):
  """Driver function that builds one visualization per episode."""
  for episode_i in range(FLAGS.num_episodes):
    xm_series = None
    if work_unit:
      xm_series = work_unit.get_measurement_series('episode_{}'.format(
          episode_i))
    with tf.Graph().as_default():
      with tf.Session():
        build_one_trajectory(episode_i, xm_series)


def draw_square(line,
                col,
                output_buffer,
                cell_size,
                square_size,
                color = None):
  """Draws a square at the given (line,col) with the give size.

  For instance, square_size=cell_size allows drawing cells of the DMLab maze.
  Using a smaller square_size is used to draw a trajectory.

  Args:
    line: Row (in DMLab maze) where to draw the square. This can be a float.
    col: Column (in DMLab maze) where to draw the square. This can be a float.
    output_buffer: np.ndarray to which the square is drawn.
    cell_size: Cell size in pixel of a DMLab maze cell.
    square_size: Size in pixel of the square to draw.
    color: The color of the square to draw given as (R, G, B) numpy array.
      Defaults to black.
  """
  if color is None:
    color = np.array([0, 0, 0], dtype=np.uint8)
  padding = square_size // 2
  half_cell_size = cell_size // 2
  row_start = int(line * cell_size + half_cell_size - padding)
  rows = np.maximum([0, 0], [row_start, row_start + square_size])
  col_start = int(col * cell_size + half_cell_size - padding)
  cols = np.minimum(output_buffer.shape[0:2],
                    [col_start, col_start + square_size])
  output_buffer[slice(*rows), slice(*cols), :] = color


def to_grid_pos(x, y, maze_size):
  """Converts DMLab world coordinates to maze coordinates.

  Inspired from LuaMazeGeneration::FromWorldPos, with the big difference that
  FromWorldPos uses (LUA) 1-based indexing.

  Args:
    x: X coordinate in DMLab world.
    y: Y coordinate in DMLab world.
    maze_size: Size of the DMLab maze in number of cell.

  Returns:
    Pair of floats: (row, col). We return floats because some uses of that
    function need the exact position in maze coordinates. Other uses are free
    to round the returned values.
  """
  grid_width = 100
  row = maze_size - y / grid_width - 0.5
  col = x / grid_width - 0.5
  return row, col


def process_env_frame(observation, reward):
  """Processes an env observation, adding color depending on the reward."""
  if FLAGS.visualization_type == 'observation':
    return observation
  observation = observation.copy()
  h = int(12 / 84 * observation.shape[0])
  w1 = int(24 / 84 * observation.shape[1])
  w2 = int(68 / 84 * observation.shape[1])
  if reward > 0:
    observation[0:h, w1:w2, :] = _GREEN
  else:
    observation[0:h, w1:w2, :] = _RED
  return observation


def build_video(trajectory_visualizations, episode_buffer,
                memory_visualizations, visualization_type, output_path):
  """Creates and saves a video from the given frames."""
  video_frames = []
  for agent, obs, memory in zip(trajectory_visualizations, episode_buffer,
                                memory_visualizations):
    if visualization_type == 'surrogate_reward':
      obs_frame = skimage.transform.resize(obs[0], [
          LEVEL_MAZE_SIZE * CELL_SIZE_PIXELS, LEVEL_MAZE_SIZE * CELL_SIZE_PIXELS
      ], mode='constant', preserve_range=True)
      video_frame = np.concatenate((agent, memory, obs_frame), axis=1)
    else:
      video_frame = obs[0]
    video_frames.append(video_frame)
  ec_logging.save_episode_buffer_as_video(video_frames, output_path)


def create_image(memory,
                 info,
                 reward,
                 previous_image = None,
                 plot_maze = True,
                 plot_agent = True,
                 plot_memory = False):
  """Creates an image for the visualization.

  Args:
    memory: Current episodic memory.
    info: dict of info from the env.
    reward: Reward from the env.
    previous_image: If not none, we draw on this image instead of starting with
      a blank image.
    plot_maze: If true, plot the DMLab maze.
    plot_agent: If true, plot a colored square (according to reward) at the
      current position of the agent.
    plot_memory: If true, plot a square at each location stored in the memory.

  Returns:
    The plotted image.
  """
  cell_size = CELL_SIZE_PIXELS
  output_size = cell_size * LEVEL_MAZE_SIZE
  layout = info.get('maze_layout')
  if layout is None:
    return np.tile(_WHITE, [output_size, output_size, 1])
  maze_size = layout.index('\n')
  assert maze_size == LEVEL_MAZE_SIZE

  if previous_image is None:
    image = np.tile(_WHITE, [output_size, output_size, 1])
  else:
    image = np.copy(previous_image)

  if plot_maze:
    for i, cell in enumerate(layout.replace('\n', '')):
      if cell != '*':
        continue
      line = i // maze_size
      col = i % maze_size
      draw_square(line, col, image, cell_size, cell_size)

  if plot_agent:
    agent_row, agent_col = to_grid_pos(info['position'][0], info['position'][1],
                                       maze_size)
    if reward > 0:
      color = np.array([0, 255, 0], dtype=np.uint8)
    else:
      color = np.array([255, 0, 0], dtype=np.uint8)

    # Instead of dot, could draw a cone with the viewing angle.
    draw_square(
        agent_row,
        agent_col,
        image,
        cell_size,
        CELL_SIZE_PIXELS // 4,
        color=color)

  if plot_memory:
    for memory_info in memory.info_memory:
      if memory_info is None:
        continue
      pos = memory_info['position']
      row, col = to_grid_pos(pos[0], pos[1], maze_size)
      color = np.array([0, 0, 255], dtype=np.uint8)
      # Instead of square, we could imaging drawing a cone with the viewing
      # angle.
      draw_square(
          row, col, image, cell_size, CELL_SIZE_PIXELS // 4, color=color)

  return image


def build_one_trajectory(episode_i=0, xm_series=None):
  """Builds the visualization for one episode. Saves files to workdir."""
  # We re-build the env to make sure we control the seed.
  num_envs = 1
  env, _, _ = env_factory.create_environments(
      LEVEL_NAME,
      num_envs,
      FLAGS.r_net_weights,
      FLAGS.dmlab_homepath,
      action_set=FLAGS.action_set,
      base_seed=FLAGS.base_env_seed + episode_i,
      environment_engine=FLAGS.environment_engine)

  trajectory_filename = os.path.join(
      FLAGS.workdir, TRAJECTORY_FILENAME_TEMPLATE.format(episode_i))

  policy = load_policy(FLAGS.policy_path, env)

  policy_state_dim = 512
  policy_states = np.zeros((num_envs, policy_state_dim), dtype=np.float32)

  np.random.seed(FLAGS.numpy_random_seed)
  observations = env.reset()
  # One frame per timestep, representing the trajectory color according to
  # the reward.
  trajectory_visualizations = []
  # One frame per timestep, showing the state of memory.
  memory_visualizations = []
  # One (env_frame, env_info) per timestep.
  episode_buffer = []
  # One action for each time step.
  saved_actions = []
  if FLAGS.trajectory_mode == 'load':
    with open(trajectory_filename, 'r') as trajectory_file:
      saved_actions = pickle.load(trajectory_file)

  dones = [False] * num_envs
  for step_i in range(FLAGS.num_steps):
    logging.info('STEP: %d / %d', step_i, FLAGS.num_steps)
    resized = [
        curiosity_env_wrapper.resize_observation(obs, [84, 84, 3])
        for obs in observations
    ] if len(observations[0].shape) >= 3 else observations
    if FLAGS.trajectory_mode != 'load':
      actions, _, policy_states, _ = policy(resized, policy_states, dones)
      if FLAGS.trajectory_mode == 'save':
        saved_actions.append(actions[0])
    else:
      actions = [saved_actions[step_i]]

    if np.random.uniform(0, 1) < FLAGS.eps_degraded:
      assert isinstance(env.action_space, gym.spaces.Discrete), (
          '--eps_degraded>0 not supported for non-discrete action spaces')
      actions = [np.random.randint(env.action_space.n)]

    observations, rewards, dones, infos = env.step(actions)
    observation = observations[0]
    reward = rewards[0]
    if xm_series:
      xm_series.create_measurement(reward, step_i)
    done = dones[0]
    info = infos[0]
    if done:
      break
    memory = env.get_episodic_memory(0)
    if not trajectory_visualizations:
      trajectory_visualizations.append(
          create_image(memory, info, reward))
    else:
      trajectory_visualizations.append(
          create_image(
              memory,
              info,
              reward,
              previous_image=trajectory_visualizations[-1],
              plot_maze=False))
    memory_visualizations.append(
        create_image(
            memory, info, reward, plot_agent=False, plot_memory=True))
    episode_buffer.append(
        (process_env_frame(utils.get_frame(observation, info), reward), info))

  build_video(
      trajectory_visualizations, episode_buffer, memory_visualizations,
      FLAGS.visualization_type,
      os.path.join(FLAGS.workdir, VIDEO_FILENAME_TEMPLATE.format(episode_i)))
  if FLAGS.trajectory_mode == 'save':
    with open(trajectory_filename, 'w') as out_file:
      pickle.dump(saved_actions, out_file)
  env.close()


def make_model(env):
  """Creates a ppo2.Model (weights are random.)."""
  policy = {'cnn': policies.CnnPolicy,
            'lstm': policies.LstmPolicy,
            'lnlstm': policies.LnLstmPolicy,
            'mlp': policies.MlpPolicy}[FLAGS.policy_architecture]
  return ppo2.Model(
      policy=policy,
      ob_space=env.observation_space,
      ac_space=env.action_space,
      nbatch_act=1,
      nbatch_train=1,
      nsteps=1,
      ent_coef=0.01,
      vf_coef=0.5,
      max_grad_norm=0.5,
      use_curiosity=FLAGS.use_curiosity,
      curiosity_strength=FLAGS.curiosity_strength,
      forward_inverse_ratio=FLAGS.forward_inverse_ratio,
      curiosity_loss_strength=FLAGS.curiosity_loss_strength,
      random_state_predictor=False)


def load_policy(model_path, env):
  """Loads a trained policy from a checkpoint.

  Args:
    model_path: Path to the checkpoint.
    env: DMLab environment.

  Returns:
    act_fn: Action function that takes
            (observation, previous_policy_state, done_mask) and returns
            (action, vf, new_policy_state, negative log prob)
  """
  model = make_model(env)
  logging.info('ppo2.Model is built')
  logging.info('Loading pp2.Model from checkpoint: %s', model_path)
  model.load(model_path)
  logging.info('Loaded pp2.Model from checkpoint')

  act_fn = model.step
  return act_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not tf.gfile.Exists(FLAGS.workdir):
    tf.gfile.MakeDirs(FLAGS.workdir)
  utils.dump_flags_to_file(os.path.join(FLAGS.workdir, 'flags.txt'))
  gin.bind_parameter('CuriosityEnvWrapper.scale_task_reward', 0.)
  gin.bind_parameter('CuriosityEnvWrapper.scale_surrogate_reward', 1.)
  gin.parse_config_files_and_bindings(None,
                                      FLAGS.gin_bindings)
  # Hardware crashes with:
  # Failed to open library!
  # dlopen: cannot load any more object with static TLS
  FLAGS.renderer = 'software'

  work_unit = None

  visualize_curiosity_reward(work_unit)
  with tf.gfile.GFile(os.path.join(FLAGS.workdir, 'gin_config.txt'), 'w') as f:
    f.write(gin.operative_config_str())


if __name__ == '__main__':
  app.run(main)
