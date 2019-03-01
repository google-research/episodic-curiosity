# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import os.path as osp
import time
import dill
from third_party.baselines import logger
from third_party.baselines.common import explained_variance
from third_party.baselines.common.input import observation_input
from third_party.baselines.common.runners import AbstractEnvRunner
from third_party.baselines.ppo2 import pathak_utils
import numpy as np
import tensorflow as tf


class Model(object):

  def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               nsteps, ent_coef, vf_coef, max_grad_norm, use_curiosity,
               curiosity_strength, forward_inverse_ratio,
               curiosity_loss_strength, random_state_predictor):
    sess = tf.get_default_session()

    act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps,
                         reuse=True)

    if use_curiosity:
      hidden_layer_size = 256
      self.state_encoder_net = tf.make_template(
          'state_encoder_net', pathak_utils.universeHead,
          create_scope_now_=True,
          trainable=(not random_state_predictor))
      self.icm_forward_net = tf.make_template(
          'icm_forward', pathak_utils.icm_forward_model,
          create_scope_now_=True, num_actions=ac_space.n,
          hidden_layer_size=hidden_layer_size)
      self.icm_inverse_net = tf.make_template(
          'icm_inverse', pathak_utils.icm_inverse_model,
          create_scope_now_=True, num_actions=ac_space.n,
          hidden_layer_size=hidden_layer_size)
    else:
      self.state_encoder_net = None
      self.icm_forward_net = None
      self.icm_inverse_net = None

    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])
    # When computing intrinsic reward a different batch size is used (number
    # of parallel environments), thus we need to define separate
    # placeholders for them.
    X_NEXT, _ = observation_input(ob_space, nbatch_train)
    X_INTRINSIC_NEXT, _ = observation_input(ob_space, nbatch_act)
    X_INTRINSIC_CURRENT, _ = observation_input(ob_space, nbatch_act)

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                               - CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    vf_losses2 = tf.square(vpredclipped - R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio,
                                         1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0),
                                                     CLIPRANGE)))
    curiosity_loss = self.compute_curiosity_loss(
        use_curiosity, train_model.X, A, X_NEXT,
        forward_inverse_ratio=forward_inverse_ratio,
        curiosity_loss_strength=curiosity_loss_strength)
    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + curiosity_loss

    if use_curiosity:
      encoded_time_step = self.state_encoder_net(X_INTRINSIC_CURRENT)
      encoded_next_time_step = self.state_encoder_net(X_INTRINSIC_NEXT)
      intrinsic_reward = self.curiosity_forward_model_loss(
          encoded_time_step, A, encoded_next_time_step)
      intrinsic_reward = intrinsic_reward * curiosity_strength

    with tf.variable_scope('model'):
      params = tf.trainable_variables()
    # For whatever reason Pathak multiplies the loss by 20.
    pathak_multiplier = 20 if use_curiosity else 1
    grads = tf.gradients(loss * pathak_multiplier, params)
    if max_grad_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)

    def getIntrinsicReward(curr, next_obs, actions):
      return sess.run(intrinsic_reward, {X_INTRINSIC_CURRENT: curr,
                                         X_INTRINSIC_NEXT: next_obs,
                                         A: actions})
    def train(lr, cliprange, obs, next_obs, returns, masks, actions, values,
              neglogpacs, states=None):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs,
                OLDVPRED: values, X_NEXT: next_obs}
      if states is not None:
        td_map[train_model.S] = states
        td_map[train_model.M] = masks
      return sess.run(
          [pg_loss, vf_loss, entropy, approxkl, clipfrac, curiosity_loss,
           _train],
          td_map
      )[:-1]
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                       'approxkl', 'clipfrac', 'curiosity_loss']

    def save(save_path):
      ps = sess.run(params)
      with tf.gfile.Open(save_path, 'wb') as fh:
        fh.write(dill.dumps(ps))

    def load(load_path):
      with tf.gfile.Open(load_path, 'rb') as fh:
        val = fh.read()
        loaded_params = dill.loads(val)
      restores = []
      for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
      sess.run(restores)
      # If you want to load weights, also save/load observation scaling inside
      # VecNormalize

    self.getIntrinsicReward = getIntrinsicReward
    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.value = act_model.value
    self.initial_state = act_model.initial_state
    self.save = save
    self.load = load
    tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101

  def curiosity_forward_model_loss(self, encoded_state, action,
                                   encoded_next_state):
    pred_next_state = self.icm_forward_net(encoded_state, action)
    forward_loss = 0.5 * tf.reduce_mean(
        tf.squared_difference(pred_next_state, encoded_next_state), axis=1)
    forward_loss = forward_loss * 288.0
    return forward_loss

  def curiosity_inverse_model_loss(self, encoded_states, actions,
                                   encoded_next_states):
    pred_action_logits = self.icm_inverse_net(encoded_states,
                                              encoded_next_states)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_action_logits, labels=actions), name='invloss')

  def compute_curiosity_loss(self, use_curiosity, time_steps, actions,
                             next_time_steps, forward_inverse_ratio,
                             curiosity_loss_strength):
    if use_curiosity:
      with tf.name_scope('curiosity_loss'):
        encoded_time_steps = self.state_encoder_net(time_steps)
        encoded_next_time_steps = self.state_encoder_net(next_time_steps)

        inverse_loss = self.curiosity_inverse_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = self.curiosity_forward_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = tf.reduce_mean(forward_loss)

        total_curiosity_loss = curiosity_loss_strength * (
            forward_inverse_ratio * forward_loss +
            (1 - forward_inverse_ratio) * inverse_loss)
    else:
      total_curiosity_loss = tf.constant(0.0, dtype=tf.float32,
                                         name='curiosity_loss')

    return total_curiosity_loss


class Runner(AbstractEnvRunner):

  def __init__(self, env, model, nsteps, gamma, lam, eval_callback=None):
    super(Runner, self).__init__(env=env, model=model, nsteps=nsteps)
    self.lam = lam
    self.gamma = gamma

    self._eval_callback = eval_callback
    self._collection_iteration = 0

  def run(self):
    if self._eval_callback:
      global_step = (self._collection_iteration *
                     self.env.num_envs *
                     self.nsteps)
      self._eval_callback(self.model.step, global_step)

    self._collection_iteration += 1

    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
    mb_neglogpacs, mb_next_obs = [], []
    mb_states = self.states
    epinfos = []
    for _ in range(self.nsteps):
      actions, values, self.states, neglogpacs = self.model.step(self.obs,
                                                                 self.states,
                                                                 self.dones)
      mb_obs.append(self.obs.copy())
      mb_actions.append(actions)
      mb_values.append(values)
      mb_neglogpacs.append(neglogpacs)
      mb_dones.append(self.dones)
      self.obs[:], rewards, self.dones, infos = self.env.step(actions)
      mb_next_obs.append(self.obs.copy())

      if self.model.state_encoder_net:
        intrinsic_reward = self.model.getIntrinsicReward(
            mb_obs[-1], mb_next_obs[-1], actions)
        # Clip to [-1, 1] range intrinsic reward.
        intrinsic_reward = [
            max(min(x, 1.0), -1.0) for x in intrinsic_reward]
        rewards += intrinsic_reward

      for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfos.append(maybeepinfo)
      mb_rewards.append(rewards)
    # batch of steps to batch of rollouts
    mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
    mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self.model.value(self.obs, self.states, self.dones)
    # discount/bootstrap off value fn
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):
      if t == self.nsteps - 1:
        nextnonterminal = 1.0 - self.dones
        nextvalues = last_values
      else:
        nextnonterminal = 1.0 - mb_dones[t+1]
        nextvalues = mb_values[t+1]
      delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal -
               mb_values[t])
      mb_advs[t] = lastgaelam = (delta + self.gamma * self.lam *
                                 nextnonterminal * lastgaelam)
    mb_returns = mb_advs + mb_values
    return (map(sf01, (mb_obs, mb_next_obs, mb_returns, mb_dones,
                       mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


def sf01(arr):
  """Swap and then flatten axes 0 and 1.
  """
  s = arr.shape
  return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
  def f(_):
    return val
  return f


def learn(policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, train_callback=None,
          eval_callback=None, cloud_sync_callback=None, cloud_sync_interval=1000,
          workdir='', use_curiosity=False, curiosity_strength=0.01,
          forward_inverse_ratio=0.2, curiosity_loss_strength=10,
          random_state_predictor=False):
  if isinstance(lr, float):
    lr = constfn(lr)
  else:
    assert callable(lr)
  if isinstance(cliprange, float):
    cliprange = constfn(cliprange)
  else:
    assert callable(cliprange)
  total_timesteps = int(total_timesteps)

  nenvs = env.num_envs
  ob_space = env.observation_space
  ac_space = env.action_space
  nbatch = nenvs * nsteps
  nbatch_train = nbatch // nminibatches

  # pylint: disable=g-long-lambda
  make_model = lambda: Model(policy=policy, ob_space=ob_space,
                             ac_space=ac_space, nbatch_act=nenvs,
                             nbatch_train=nbatch_train, nsteps=nsteps,
                             ent_coef=ent_coef, vf_coef=vf_coef,
                             max_grad_norm=max_grad_norm,
                             use_curiosity=use_curiosity,
                             curiosity_strength=curiosity_strength,
                             forward_inverse_ratio=forward_inverse_ratio,
                             curiosity_loss_strength=curiosity_loss_strength,
                             random_state_predictor=random_state_predictor)
  # pylint: enable=g-long-lambda
  if save_interval and workdir:
    with tf.gfile.Open(osp.join(workdir, 'make_model.pkl'), 'wb') as fh:
      fh.write(dill.dumps(make_model))
  model = make_model()
  if load_path is not None:
    model.load(load_path)
  runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                  eval_callback=eval_callback)

  epinfobuf = deque(maxlen=100)
  tfirststart = time.time()

  nupdates = total_timesteps//nbatch
  for update in range(1, nupdates+1):
    assert nbatch % nminibatches == 0
    nbatch_train = nbatch // nminibatches
    tstart = time.time()
    frac = 1.0 - (update - 1.0) / nupdates
    lrnow = lr(frac)
    cliprangenow = cliprange(frac)
    (obs, next_obs, returns, masks, actions, values,
     neglogpacs), states, epinfos = runner.run()
    epinfobuf.extend(epinfos)
    mblossvals = []
    if states is None:  # nonrecurrent version
      inds = np.arange(nbatch)
      for _ in range(noptepochs):
        np.random.shuffle(inds)
        for start in range(0, nbatch, nbatch_train):
          end = start + nbatch_train
          mbinds = inds[start:end]
          slices = [arr[mbinds] for arr in (obs, returns, masks, actions,
                                            values, neglogpacs, next_obs)]
          mblossvals.append(model.train(lrnow, cliprangenow, slices[0],
                                        slices[6], slices[1], slices[2],
                                        slices[3], slices[4], slices[5]))
    else:  # recurrent version
      assert nenvs % nminibatches == 0
      envsperbatch = nenvs // nminibatches
      envinds = np.arange(nenvs)
      flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
      envsperbatch = nbatch_train // nsteps
      for _ in range(noptepochs):
        np.random.shuffle(envinds)
        for start in range(0, nenvs, envsperbatch):
          end = start + envsperbatch
          mbenvinds = envinds[start:end]
          mbflatinds = flatinds[mbenvinds].ravel()
          slices = [arr[mbflatinds] for arr in (obs, returns, masks, actions,
                                                values, neglogpacs, next_obs)]
          mbstates = states[mbenvinds]
          mblossvals.append(model.train(lrnow, cliprangenow, slices[0],
                                        slices[6], slices[1], slices[2],
                                        slices[3], slices[4], slices[5],
                                        mbstates))

    lossvals = np.mean(mblossvals, axis=0)
    tnow = time.time()
    fps = int(nbatch / (tnow - tstart))
    if update % log_interval == 0 or update == 1:
      ev = explained_variance(values, returns)
      logger.logkv('serial_timesteps', update*nsteps)
      logger.logkv('nupdates', update)
      logger.logkv('total_timesteps', update*nbatch)
      logger.logkv('fps', fps)
      logger.logkv('explained_variance', float(ev))
      logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
      logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
      if train_callback:
        train_callback(safemean([epinfo['l'] for epinfo in epinfobuf]),
                       safemean([epinfo['r'] for epinfo in epinfobuf]),
                       update * nbatch)
      logger.logkv('time_elapsed', tnow - tfirststart)
      for (lossval, lossname) in zip(lossvals, model.loss_names):
        logger.logkv(lossname, lossval)
      logger.dumpkvs()
    if (save_interval and (update % save_interval == 0 or update == 1) and
        workdir):
      checkdir = osp.join(workdir, 'checkpoints')
      if not tf.gfile.Exists(checkdir):
        tf.gfile.MakeDirs(checkdir)
      savepath = osp.join(checkdir, '%.5i'%update)
      print('Saving to', savepath)
      model.save(savepath)
    if (cloud_sync_interval and update % cloud_sync_interval == 0 and
        cloud_sync_callback):
      cloud_sync_callback()
  env.close()
  return model


def safemean(xs):
  return np.mean(xs) if xs else np.nan
