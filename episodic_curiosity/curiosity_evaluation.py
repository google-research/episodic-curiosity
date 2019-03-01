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

"""Library to evaluate exploration."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from episodic_curiosity.constants import Const
import numpy as np
import tensorflow as tf


class OracleExplorationReward(object):
  """Class that computes the ideal exploration bonus."""

  def __init__(self, reward_grid_size):
    """Creates a new oracle to compute the exploration reward."""
    self._reward_grid_size = reward_grid_size
    self._collected_positions = set()

  def update_state(self, agent_position):
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
      return 1.0


def policy_state_coverage(env, policy_action, reward_grid_size,
                          eval_time_steps):
  """Computes the maze coverage by a given policy.

  Args:
    env: A Gym environment.
    policy_action: Function which, given a state, returns an action.
      e.g. For DMLab, the policy will return an Actions protobuf.
    reward_grid_size: L1 distance between 2 consecutive curiosity reward.
    eval_time_steps: List of times after which state coverage should be
      computed.

  Returns:
    The total number of cumulative rewards at different times
    in the episode.
  """
  max_episode_length = max(eval_time_steps) + 1

  # During test, the exploration reward is given by an oracle that has
  # access to the agent coordinates.
  cumulative_exploration_reward = 0.0
  oracle_exploration_reward = OracleExplorationReward(
      reward_grid_size=reward_grid_size)

  # Initial observation.
  observation = env.reset()

  reward_list = {}
  for k in range(max_episode_length):
    action = policy_action(observation)

    # TODO(damienv): Repeating an action should be a wrapper
    # of the environment.
    repeat_action_count = Const.ACTION_REPEAT
    done = False
    for _ in range(repeat_action_count):
      observation, _, done, metadata = env.step(action)
      if done:
        break

    # Abort if we've reached the end of the episode.
    if done:
      reward_list[k] = cumulative_exploration_reward
      break

    # Convert the new agent position into a possible exploration bonus.
    # Note: getting the position of the agent is specific to DMLab.
    agent_position = metadata['position']
    cumulative_exploration_reward += oracle_exploration_reward.update_state(
        agent_position)

    step_count = k + 1
    if step_count in eval_time_steps:
      reward_list[step_count] = cumulative_exploration_reward

  return reward_list


class PolicyWrapper(object):
  """Wrap a policy defined by some input/output nodes in a TF graph."""

  def __init__(self,
               input_observation,
               input_state,
               output_state,
               output_actions):
    # The tensors to feed the policy and to retrieve the relevant outputs.
    self._input_observation = input_observation
    self._input_state = input_state
    self._output_state = output_state
    self._output_actions = output_actions

    # TensorFlow session.
    self._sess = None

  def set_session(self, tf_session):
    self._sess = tf_session

  def reset(self):
    self._current_state = np.zeros(
        self._input_state.get_shape().as_list(),
        dtype=np.float32)

  def action(self, observation):
    """Action to perform given an observation."""
    # Converts to batched obervation (with batch_size=1).
    observation = np.expand_dims(observation, axis=0)

    # Run the underlying policy and update the state of the policy.
    actions, next_state = self._sess.run(
        [self._output_actions, self._output_state],
        feed_dict={self._input_observation: observation,
                   self._input_state: self._current_state})
    self._current_state = next_state

    # Un-batch the action.
    action = actions[0]
    return action


def load_policy(graph_def,
                input_observation_name,
                input_state_name,
                output_state_name,
                output_pd_params_name,
                tf_sampling_fn):
  """Load a policy from a graph file.

  Args:
    graph_def: Graph definition.
    input_observation_name: Name in the graph definition of the tensor
      of observations.
    input_state_name: Name in the graph definition of the tensor
      of input states.
    output_state_name: Name in the graph definition of the tensor of output
      states.
    output_pd_params_name: Name in the graph definition of the tensor
      representing the parameters of the distribution over actions.
    tf_sampling_fn: Function which samples action based in the probability
      distribution parameters (given by output_pd_params_name).

  Returns:
    Returns a python policy.
  """
  tensors = tf.import_graph_def(
      graph_def,
      return_elements=[input_observation_name,
                       input_state_name,
                       output_state_name,
                       output_pd_params_name],
      name='')

  input_observation, input_state, output_state, output_pd_params = tensors
  output_actions = tf_sampling_fn(output_pd_params)

  return PolicyWrapper(input_observation,
                       input_state,
                       output_state,
                       output_actions)
