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

"""R-network and some related functions to train R-networks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tempfile

from absl import logging
from third_party.keras_resnet import models
import numpy as np
import tensorflow as tf
from tensorflow import keras


class RNetwork(object):
  """Encapsulates a trained R network, with lazy loading of weights."""

  def __init__(self, input_shape, weight_path):
    """Inits the RNetwork.

    Args:
      input_shape: (height, width, channel)
      weight_path: Path to the weights of the r_network.
    """
    self._weight_path = weight_path
    (self._r_network, self._embedding_network,
     self._similarity_network) = models.ResnetBuilder.build_siamese_resnet_18(
         input_shape)
    self._r_network.compile(
        loss='categorical_crossentropy', optimizer=keras.optimizers.Adam())
    self._weights_loaded = False

  def _maybe_load_weights(self):
    """Loads R-network weights if needed.

    The RNetwork is used together with an environment used by ppo2.learn.
    Unfortunately, ppo2.learn initializes all global TF variables at the
    beginning of the training, which in particular, random-initializes the
    weights of the R Network. We therefore load the weights lazily, to make sure
    they are loaded after the global initialization happens in ppo2.learn.
    """
    if self._weights_loaded:
      return
    if self._weight_path is None:
      # Typically the case when doing online training of the R-network.
      return
    # Keras does not support reading weights from CNS, so we have to copy the
    # weights to a temporary local file.
    with tempfile.NamedTemporaryFile(prefix='r_net', suffix='.h5',
                                     delete=False) as tmp_file:
      tmp_path = tmp_file.name
    tf.gfile.Copy(self._weight_path, tmp_path, overwrite=True)
    logging.info('Loading weights from %s...', tmp_path)
    print('Loading into R network:')
    self._r_network.summary()
    self._r_network.load_weights(tmp_path)
    tf.gfile.Remove(tmp_path)
    self._weights_loaded = True

  def embed_observation(self, x):
    """Embeds an observation.

    Args:
      x: batched input observations. Expected to have the shape specified when
         the RNetwork was contructed (plus the batch dimension as first dim).

    Returns:
      embedding, shape [batch, models.EMBEDDING_DIM]
    """
    self._maybe_load_weights()
    return self._embedding_network.predict(x)

  def embedding_similarity(self, x, y):
    """Computes the similarity between two embeddings.

    Args:
      x: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].
      y: batch of the first embedding. Shape [batch, models.EMBEDDING_DIM].

    Returns:
      Similarity probabilities. 1 means very similar according to the net.
      0 means very dissimilar. Shape [batch].
    """
    self._maybe_load_weights()
    return self._similarity_network.predict([x, y],
                                            batch_size=1024)[:, 1]
