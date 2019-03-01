# coding=utf-8
"""VecEnv implementation using python threads instead of subprocesses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
from third_party.baselines.common.vec_env import VecEnv
import numpy as np
from six.moves import queue as Queue  # pylint: disable=redefined-builtin


def thread_worker(send_q, recv_q, env_fn):
    """Similar to SubprocVecEnv.worker(), but for TreadedVecEnv.

    Args:
      send_q: Queue which ThreadedVecEnv sends commands to.
      recv_q: Queue which ThreadedVecEnv receives commands from.
      env_fn: Callable that creates an instance of the environment.
    """
    env = env_fn()
    while True:
        cmd, data = send_q.get()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            recv_q.put((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            recv_q.put(ob)
        elif cmd == 'render':
            recv_q.put(env.render(mode='rgb_array'))
        elif cmd == 'close':
            break
        elif cmd == 'get_spaces':
            recv_q.put((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ThreadedVecEnv(VecEnv):
    """Similar to SubprocVecEnv, but uses python threads instead of subprocs.

    Sub-processes involve forks, and a lot of code (incl. google3's) is not
    fork-safe, leading to deadlocks. The drawback of python threads is that the
    python code is still executed serially because of the GIL. However, many
    environments do the heavy lifting in C++ (where the GIL is released, and
    hence execution can happen in parallel), so python threads are not often
    limiting.
    """

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in python threads.
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.send_queues = [Queue.Queue() for _ in range(nenvs)]
        self.recv_queues = [Queue.Queue() for _ in range(nenvs)]
        self.threads = [threading.Thread(target=thread_worker,
                                         args=(send_q, recv_q, env_fn))
                        for (send_q, recv_q, env_fn) in
                        zip(self.send_queues, self.recv_queues, env_fns)]
        for thread in self.threads:
          thread.daemon = True
          thread.start()

        self.send_queues[0].put(('get_spaces', None))
        observation_space, action_space = self.recv_queues[0].get()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for send_q, action in zip(self.send_queues, actions):
            send_q.put(('step', action))
        self.waiting = True

    def step_wait(self):
        results = self._receive_all()
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._send_all(('reset', None))
        return np.stack(self._receive_all())

    def reset_task(self):
        self._send_all(('reset_task', None))
        return np.stack(self._receive_all())

    def close(self):
        if self.closed:
            return
        if self.waiting:
            self._receive_all()
        self._send_all(('close', None))
        for thread in self.threads:
            thread.join()
        self.closed = True

    def render(self, mode='human'):
        raise NotImplementedError

    def _send_all(self, item):
        for send_q in self.send_queues:
            send_q.put(item)

    def _receive_all(self):
        return [recv_q.get() for recv_q in self.recv_queues]
