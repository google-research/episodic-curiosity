# coding=utf-8
# COPYRIGHT
#
# All contributions by Raghavendra Kotikalapudi:
# Copyright (c) 2016, Raghavendra Kotikalapudi.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2016, the respective contributors.
# All rights reserved.
#
# Copyright (c) 2018 Google LLC
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Model definitions for the R-network.

Forked from https://github.com/raghakot/keras-resnet/blob/master/resnet.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
# pytype: disable=import-error
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
# pytype: enable=import-error


EMBEDDING_DIM = 512
TOP_HIDDEN = 4


def _bn_relu(inpt):
  """Helper to build a BN -> relu block."""
  norm = BatchNormalization(axis=3)(inpt)
  return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
  """Helper to build a conv -> BN -> relu block."""
  filters = conv_params["filters"]
  kernel_size = conv_params["kernel_size"]
  strides = conv_params.setdefault("strides", (1, 1))
  kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
  padding = conv_params.setdefault("padding", "same")
  kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

  def f(inpt):
    conv = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(
            inpt)
    return _bn_relu(conv)

  return f


def _bn_relu_conv(**conv_params):
  """Helper to build a BN -> relu -> conv block."""
  # This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
  filters = conv_params["filters"]
  kernel_size = conv_params["kernel_size"]
  strides = conv_params.setdefault("strides", (1, 1))
  kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
  padding = conv_params.setdefault("padding", "same")
  kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

  def f(inpt):
    activation = _bn_relu(inpt)
    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(
            activation)

  return f


def _shortcut(inpt, residual):
  """Adds shortcut between inpt and residual block and merges with "sum"."""
  # Expand channels of shortcut to match residual.
  # Stride appropriately to match residual (width, height)
  # Should be int if network architecture is correctly configured.
  input_shape = K.int_shape(inpt)
  residual_shape = K.int_shape(residual)
  stride_width = int(round(input_shape[1] / residual_shape[1]))
  stride_height = int(round(input_shape[2] / residual_shape[2]))
  equal_channels = input_shape[3] == residual_shape[3]

  shortcut = inpt
  # 1 X 1 conv if shape is different. Else identity.
  if stride_width > 1 or stride_height > 1 or not equal_channels:
    shortcut = Conv2D(
        filters=residual_shape[3],
        kernel_size=(1, 1),
        strides=(stride_width, stride_height),
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0001))(
            inpt)

  return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
  """Builds a residual block with repeating bottleneck blocks."""

  def f(inpt):
    """Helper function."""
    for i in range(repetitions):
      init_strides = (1, 1)
      if i == 0 and not is_first_layer:
        init_strides = (2, 2)
      inpt = block_function(
          filters=filters,
          init_strides=init_strides,
          is_first_block_of_first_layer=(is_first_layer and i == 0))(
              inpt)
    return inpt

  return f


def basic_block(filters,
                init_strides=(1, 1),
                is_first_block_of_first_layer=False):
  """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34."""
  # Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

  def f(inpt):
    """Helper function."""
    if is_first_block_of_first_layer:
      # don't repeat bn->relu since we just did bn->relu->maxpool
      conv1 = Conv2D(
          filters=filters,
          kernel_size=(3, 3),
          strides=init_strides,
          padding="same",
          kernel_initializer="he_normal",
          kernel_regularizer=l2(1e-4))(
              inpt)
    else:
      conv1 = _bn_relu_conv(
          filters=filters, kernel_size=(3, 3), strides=init_strides)(
              inpt)

    residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
    return _shortcut(inpt, residual)

  return f


def _bn_relu_for_dense(inpt):
  norm = BatchNormalization(axis=1)(inpt)
  return Activation("relu")(norm)


def _top_network(input_shape):
  """Add top classification layers.

  Args:
    input_shape: shape of the embedding of the input image.

  Returns:
    A model taking a batch of input image embeddings, returning a batch of
    similarities (shape [batch, 2])
  """
  x1 = Input(shape=input_shape, name="top_deep_net_x1")
  x2 = Input(shape=input_shape, name="top_deep_net_x2")
  x = concatenate([x1, x2])
  raw_result = _bn_relu_for_dense(x)
  for _ in range(TOP_HIDDEN):
    raw_result = Dense(
        units=EMBEDDING_DIM, kernel_initializer="he_normal")(
            raw_result)
    raw_result = _bn_relu_for_dense(raw_result)
  output = Dense(
      units=2, activation="softmax", kernel_initializer="he_normal")(
          raw_result)
  model = Model(inputs=[x1, x2], outputs=output)
  model.summary()
  return model


def _metric_top_network(input_shape):
  """A simple top network that basically computes sigmoid(dot_product(x1, x2)).

  Args:
    input_shape: shape of the embedding of the input image.

  Returns:
    A model taking a batch of input image embeddings, returning a batch of
    similarities (shape [batch, 2])
  """
  x1 = Input(shape=input_shape, name="top_metric_net_x1")
  x2 = Input(shape=input_shape, name="top_metric_net_x2")

  def one_hot_sigmoid(x):
    return K.concatenate([1 - sigmoid(x), sigmoid(x)], axis=1)

  dot_product = Dot(axes=-1)([x1, x2])
  output = Lambda(one_hot_sigmoid)(dot_product)
  model = Model(inputs=[x1, x2], outputs=output)
  model.summary()
  return model


class ResnetBuilder(object):
  """Factory class for creating Resnet models."""

  @staticmethod
  def build(input_shape, num_outputs, block_fn, repetitions, is_classification):
    """Builds a custom ResNet like architecture.

    Args:
      input_shape: The inpt shape in the form (nb_rows, nb_cols, nb_channels)
      num_outputs: The number of outputs at final softmax layer
      block_fn: The block function to use. This is either `basic_block` or
        `bottleneck`. The original paper used basic_block for layers < 50
      repetitions: Number of repetitions of various block units. At each block
        unit, the number of filters are doubled and the inpt size is halved
      is_classification: if True add softmax layer on top

    Returns:
      The keras `Model`.
      The model's input is an image tensor. Its shape is [batch, height, width,
      channels] if the backend is tensorflow.
      The model's output is the embedding with shape [batch, num_outputs].

    Raises:
      Exception: wrong input shape.
    """
    if len(input_shape) != 3:
      raise Exception(
          "Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")

    inpt = Input(shape=input_shape)
    conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(inpt)
    pool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding="same")(
            conv1)

    block = pool1
    filters = 64
    for i, r in enumerate(repetitions):
      block = _residual_block(
          block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(
              block)
      filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(
        pool_size=(block_shape[1], block_shape[2]),
        strides=(1, 1))(
            block)
    flatten1 = Flatten()(pool2)
    last_activation = None
    if is_classification:
      last_activation = "softmax"
    dense = Dense(
        units=num_outputs,
        kernel_initializer="he_normal",
        activation=last_activation)(
            flatten1)

    model = Model(inputs=inpt, outputs=dense)
    model.summary()
    return model

  @staticmethod
  def build_resnet_18(input_shape, num_outputs, is_classification=True):
    """Create Resnet-18."""
    return ResnetBuilder.build(input_shape, num_outputs, basic_block,
                               [2, 2, 2, 2], is_classification)

  @staticmethod
  @gin.configurable
  def build_siamese_resnet_18(input_shape,
                              use_deep_top_network=True,
                              trainable_bottom_network=True):
    """Create siamese architecture for R-network.

    Args:
      input_shape: Shape of the input images, (height, width, channels)
      use_deep_top_network: If true (default), a deep network will be used for
                            comparing embeddings. Otherwise, we use a simple
                            distance metric.
      trainable_bottom_network: Whether the bottom (embedding) model is
                                trainable.

    Returns:
      A tuple:
        - The model mapping two images [batch, height, width, channels] to
          similarities [batch, 2].
        - The embedding model mapping one image [batch, height, width, channels]
          to embedding [batch, EMBEDDING_DIM].
        - The similarity model mapping two embedded images
          [batch, 2*EMBEDDING_DIM] to similariries [batch, 2].
      The returned models share weights. In particular, loading the weights of
      the first model also loads the weights of the other two models.
    """
    branch = ResnetBuilder.build_resnet_18(
        input_shape, EMBEDDING_DIM, is_classification=False)
    branch.trainable = trainable_bottom_network
    x1 = Input(shape=input_shape, name="x1")
    x2 = Input(shape=input_shape, name="x2")
    y1 = branch(x1)
    y2 = branch(x2)
    if use_deep_top_network:
      similarity_network = _top_network((EMBEDDING_DIM,))
    else:
      similarity_network = _metric_top_network((EMBEDDING_DIM,))

    output = similarity_network([y1, y2])
    return Model(inputs=[x1, x2], outputs=output), branch, similarity_network
