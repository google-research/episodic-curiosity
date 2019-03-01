# coding=utf-8
"""Helpers for scripts like run_atari.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from third_party.baselines import logger
from third_party.baselines.bench import Monitor
from third_party.baselines.common import set_global_seeds
from third_party.baselines.common.atari_wrappers import make_atari
from third_party.baselines.common.atari_wrappers import wrap_deepmind
from third_party.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0,
                   use_monitor=True):
  """Create a wrapped, monitored SubprocVecEnv for Atari.
  """
  if wrapper_kwargs is None: wrapper_kwargs = {}
  def make_env(rank):  # pylint: disable=C0111
    def _thunk():
      env = make_atari(env_id)
      env.seed(seed + rank)
      if use_monitor:
        env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                             str(rank)))
      return wrap_deepmind(env, **wrapper_kwargs)
    return _thunk
  set_global_seeds(seed)
  return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
