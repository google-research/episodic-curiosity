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

"""Simple test of the curiosity evaluation."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from episodic_curiosity import curiosity_evaluation
from episodic_curiosity.environments import fake_gym_env
import numpy as np
import tensorflow as tf


def random_policy(unused_observation):
  action = np.random.randint(low=0, high=fake_gym_env.FakeGymEnv.NUM_ACTIONS)
  return action


class CuriosityEvaluationTest(tf.test.TestCase):

  def EvalPolicy(self, policy):
    env = fake_gym_env.FakeGymEnv()

    # Distance between 2 consecutive curiosity rewards.
    reward_grid_size = 10.0

    # Times of evaluation.
    eval_time_steps = [100, 500, 3000]

    rewards = curiosity_evaluation.policy_state_coverage(
        env, policy, reward_grid_size, eval_time_steps)

    # The exploration reward is at most the number of steps.
    # It is equal to the number of steps when the policy explores a new state
    # at every time step.
    print('Curiosity reward: {}'.format(rewards))
    for k, r in rewards.items():
      self.assertGreaterEqual(k, r)

  def testRandomPolicy(self):
    self.EvalPolicy(random_policy)

  def testNNPolicy(self):
    batch_size = 1
    x = tf.placeholder(
        tf.float32,
        shape=(batch_size,) + fake_gym_env.FakeGymEnv.OBSERVATION_SHAPE)
    x = tf.div(x, 255.0)

    # This is just to make the test run fast enough.
    x_downscaled = tf.image.resize_images(x, [8, 8])
    x_downscaled = tf.reshape(x_downscaled, [batch_size, -1])

    # Logits to select the action.
    num_actions = 7
    h = tf.contrib.layers.fully_connected(
        inputs=x_downscaled,
        num_outputs=32,
        activation_fn=None, scope='fc0')
    h = tf.nn.relu(h)
    y_logits = tf.contrib.layers.fully_connected(
        inputs=h,
        num_outputs=num_actions,
        activation_fn=None, scope='fc1')
    temperature = 100.0
    y_logits /= temperature

    # Draw the action according to the distribution inferred by the logits.
    r = tf.random_uniform(tf.shape(y_logits),
                          minval=0.001, maxval=0.999)
    y_logits -= tf.log(-tf.log(r))
    y = tf.argmax(y_logits, axis=-1)

    input_state = tf.placeholder(tf.float32, shape=(37))
    output_state = input_state

    # Policy from the previous network.
    policy = curiosity_evaluation.PolicyWrapper(
        x, input_state, output_state, y)

    global_init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(global_init_op)

      policy.set_session(sess)
      policy.reset()

      self.EvalPolicy(policy.action)


if __name__ == '__main__':
  tf.test.main()
