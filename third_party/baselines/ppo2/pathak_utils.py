# coding=utf-8
"""Utility functions used by Pathak's curiosity algorithm.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad='SAME',
           dtype=tf.float32, collections=None, trainable=True):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x = tf.to_float(x)
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
                    num_filters]

    # there are 'num input feature maps * filter height * filter width'
    # inputs to each hidden unit
    fan_in = np.prod(filter_shape[:3])
    # each unit in the lower layer receives a gradient from:
    # 'num output feature maps * filter height * filter width' /
    #   pooling size
    fan_out = np.prod(filter_shape[:2]) * num_filters
    # initialize weights with random weights
    w_bound = np.sqrt(6. / (fan_in + fan_out))

    w = tf.get_variable('W', filter_shape, dtype,
                        tf.random_uniform_initializer(-w_bound, w_bound),
                        collections=collections, trainable=trainable)
    b = tf.get_variable('b', [1, 1, 1, num_filters],
                        initializer=tf.constant_initializer(0.0),
                        collections=collections, trainable=trainable)
    return tf.nn.conv2d(x, w, stride_shape, pad) + b


def flatten(x):
  return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def normalized_columns_initializer(std=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return _initializer


def linear(x, size, name, initializer=None, bias_init=0):
  w = tf.get_variable(name + '/w', [x.get_shape()[1], size],
                      initializer=initializer)
  b = tf.get_variable(name + '/b', [size],
                      initializer=tf.constant_initializer(bias_init))
  return tf.matmul(x, w) + b


def universeHead(x, nConvs=4, trainable=True):
  """Universe agent example.

  Args:
    x: input image
    nConvs: number of convolutional layers
    trainable: whether conv2d variables are trainable

  Returns:
    [None, 288] embedding
  """
  print('Using universe head design')
  x = tf.image.resize_images(x, [42, 42])
  x = tf.cast(x, tf.float32) / 255.
  for i in range(nConvs):
    x = tf.nn.elu(conv2d(x, 32, 'l{}'.format(i + 1), [3, 3], [2, 2],
                         trainable=trainable))
    # print('Loop{} '.format(i+1),tf.shape(x))
    # print('Loop{}'.format(i+1),x.get_shape())
  x = flatten(x)
  return x


def icm_forward_model(encoded_state, action, num_actions, hidden_layer_size):
  action = tf.one_hot(action, num_actions)
  combined_input = tf.concat([encoded_state, action], axis=1)
  hidden = tf.nn.relu(linear(combined_input, hidden_layer_size, 'f1',
                             normalized_columns_initializer(0.01)))
  pred_next_state = linear(hidden, encoded_state.get_shape()[1].value, 'flast',
                           normalized_columns_initializer(0.01))
  return pred_next_state


def icm_inverse_model(encoded_state, encoded_next_state, num_actions,
                      hidden_layer_size):
  combined_input = tf.concat([encoded_state, encoded_next_state], axis=1)
  hidden = tf.nn.relu(linear(combined_input, hidden_layer_size, 'g1',
                             normalized_columns_initializer(0.01)))
  # Predicted action logits
  logits = linear(hidden, num_actions, 'glast',
                  normalized_columns_initializer(0.01))
  return logits
