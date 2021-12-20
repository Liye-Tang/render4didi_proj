#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json
import os

import tensorflow as tf
import numpy as np

from env_build.utils.policy import AttentionPolicy4Toyota
from env_build.utils.preprocessor import Preprocessor
from env_build.endtoend import CrossroadEnd2endMix
from env_build.utils.model import Net


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        self.policy = AttentionPolicy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         args=self.args, gamma=self.args.gamma)

        init_obs = np.ones((201, ), dtype=np.float32)  # TODO
        mask = np.ones((18,), dtype=np.float32)
        self.run_batch(init_obs[np.newaxis, :], mask[np.newaxis, :])
        self.obj_value_batch(init_obs[np.newaxis, :], mask[np.newaxis, :])
        self.Net = Net(self.policy.policy, self.policy.obj_v, self.policy.attn_net, name='UnifiedNet',
                       other_start_dim=21, d_obs=201)  # TODO
        self.run_batch_unified(init_obs[np.newaxis, :], mask[np.newaxis, :])

    @tf.function
    def run_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states, weights = self._get_states(processed_obses, masks)
        actions = self.policy.compute_mode(states)
        return actions, weights

    @tf.function
    def obj_value_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states, _ = self._get_states(processed_obses, masks)
        values = self.policy.compute_obj_v(states)
        return values

    def _get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state, mb_attn_weights

    def run_batch_unified(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        # print(np.concatenate([processed_obses, masks], axis=1))
        results = self.Net(np.concatenate([processed_obses, masks], axis=1))
        # print(results)

    def sava_h5_model(self):
        logdir = './saved_model'
        os.makedirs(logdir)
        # self.Net.summary()
        self.Net.save_weights('./saved_model/{}.h5'.format('net_ego'))

    def test_model(self):
        obs = np.concatenate([1 * np.ones((1, 10), dtype=np.float32), 2 * np.ones((1, 191), dtype=np.float32)], axis=1)
        mask = np.concatenate([np.ones((1, 10), dtype=np.float32), np.ones((1, 8), dtype=np.float32)], axis=1)
        # obs = obs[np.newaxis, :]
        # mask = obs[np.newaxis, :]
        obs_others, mb_attn_weights = self.policy.compute_attn(obs[:, 21:], mask)
        obs = np.concatenate((obs[:, :21], obs_others), axis=1)
        logits = self.policy.policy(obs)
        action, _ = tf.split(logits, num_or_size_splits=2, axis=-1)
        action = tf.tanh(action)
        value = self.policy.obj_v(obs)
        result = self.Net(tf.concat([1 * tf.ones((1, 10), dtype=tf.float32), 2 * tf.ones((1, 191), dtype=tf.float32), tf.ones((1, 10), dtype=tf.float32),tf.ones((1, 8), dtype=tf.float32)], axis=1))
        print(tf.concat((value, action), axis=1))
        print(result)


if __name__ == '__main__':
    a = LoadPolicy('../utils/models/{}'.format('experiment-2021-12-11-16-53-29'), 135000)
    # a.test_model()
    a.sava_h5_model()