#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import shutil
import time
import json
import os
import heapq
from math import cos, sin, pi

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import numpy as np
import tensorflow as tf

from env_build.dynamics_and_models import EnvironmentModel, ReferencePath
from env_build.endtoend import CrossroadEnd2endMix
from env_build.endtoend_env_utils import *
from multi_path_generator import MultiPathGenerator
from env_build.utils.load_policy import LoadPolicy
from env_build.utils.misc import TimerStat
from env_build.utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, train_exp_dir, ite, logdir=None):
        self.policy = LoadPolicy('../utils/models/{}'.format(train_exp_dir), ite)
        self.args = self.policy.args
        self.env = CrossroadEnd2endMix(mode='testing', future_point_num=self.args.num_rollout_list_for_policy_update[0])
        self.model = EnvironmentModel(mode='testing')
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig_plot = 0
        self.hist_posi = []
        self.old_index = 0
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        # ------------------build graph for tf.function in advance-----------------------
        obs, all_info = self.env.reset()
        mask, future_n_point = all_info['mask'], all_info['future_n_point']
        obs = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.float32)
        mask = tf.convert_to_tensor(mask[np.newaxis, :], dtype=tf.float32)
        future_n_point = tf.convert_to_tensor(future_n_point[np.newaxis, :], dtype=tf.float32)
        self.is_safe(obs, mask, future_n_point)
        self.policy.run_batch(obs, mask)
        self.policy.obj_value_batch(obs, mask)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset()

    def reset(self,):
        self.obs, _ = self.env.reset()
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        self.recorder.reset()
        self.old_index = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
            if self.episode_counter >= 1:
                select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                                                               self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                                                               isshow=False)
        return self.obs

    @tf.function
    def is_safe(self, obses, masks, future_n_point):
        self.model.reset(obses, future_n_point)
        punish = 0.
        for step in range(5):
            action, _ = self.policy.run_batch(obses, masks)
            obses, _, _, _, _, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, _ \
                = self.model.rollout_out(action)
            punish += veh2veh4real[0] + veh2bike4real[0] + veh2person4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, real_mask, real_future_n_point):
        action_safe_set = [[[0., -1.]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        real_mask = tf.convert_to_tensor(real_mask[np.newaxis, :], dtype=tf.float32)
        real_future_n_point = tf.convert_to_tensor(real_future_n_point[np.newaxis, :], dtype=tf.float32)
        if not self.is_safe(real_obs, real_mask, real_future_n_point):
            print('SAFETY SHIELD STARTED!')
            _, weight = self.policy.run_batch(real_obs, real_mask)
            return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), weight.numpy()[0], True
        else:
            action, weight = self.policy.run_batch(real_obs, real_mask)
            return action.numpy()[0], weight.numpy()[0], False

    @tf.function
    def cal_prediction_obs(self, real_obs, real_mask, real_future_n_point):
        obses, masks = real_obs[np.newaxis, :], real_mask[np.newaxis, :]
        ref_points = tf.expand_dims(real_future_n_point, axis=0)
        self.model.reset(obses)
        obses_list = []
        for i in range(25):
            action, _ = self.policy.run_batch(obses, masks)
            obses, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                veh2person4real, _, reward_dict = self.model.rollout_out_online(action, ref_points[:, :, i])
            obses_list.append(obses[0])
        return tf.convert_to_tensor(obses_list)

    def step(self):
        self.step_counter += 1
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        with self.step_timer:
            obs_list, mask_list, future_n_point_list = [], [], []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                vector, mask_vector, future_n_point = self.env._get_obs()
                obs_list.append(vector)
                mask_list.append(mask_vector)
                future_n_point_list.append(future_n_point)
            all_obs = tf.stack(obs_list, axis=0).numpy()
            all_mask = tf.stack(mask_list, axis=0).numpy()

            path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
            old_value = path_values[self.old_index]
            # value is to approximate (- sum of reward)
            new_index, new_value = int(np.argmin(path_values)), min(path_values)
            # rule for equal traj value
            path_index_error = []
            if self.step_counter % 3 == 0:
                if heapq.nsmallest(2, path_values)[0] == heapq.nsmallest(2, path_values)[1]:
                    for i in range(len(path_values)):
                        if path_values[i] == min(path_values):
                            index_error = abs(self.old_index - i)
                            path_index_error.append(index_error)
                    # new_index_new = min(path_index_error) + self.old_index if min(path_index_error) + self.old_index < 4 else self.old_index - min(path_index_error)
                    new_index_new = self.old_index - min(path_index_error) if self.old_index - min(path_index_error) > -1 else self.old_index + min(path_index_error)
                    new_value_new = path_values[new_index_new]
                    path_index = self.old_index if old_value - new_value_new < 0.01 else new_index_new
                else:
                    path_index = self.old_index if old_value - new_value < 0.01 else new_index
                self.old_index = path_index
            else:
                path_index = self.old_index
            self.env.set_traj(self.path_list[path_index])
            obs_real, mask_real, future_n_point_real = obs_list[path_index], mask_list[path_index], future_n_point_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, weights, is_ss = self.safe_shield(obs_real, mask_real, future_n_point_real)
            # print('ALL TIME:', self.step_timer.mean, 'ss', self.ss_timer.mean)

        # obses, masks = obs_real[np.newaxis, :], mask_real[np.newaxis, :]
        # ref_points = tf.expand_dims(future_n_point_real, axis=0)
        # self.model.reset(obses)
        # obses_list = []
        # for i in range(25):
        #     action = self.policy.run_batch(obses, masks)
        #     obses, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
        #         veh2person4real, reward_dict = self.model.rollout_out_online(action, ref_points[:, :, i])
        #     obses_list.append(obses[0])
        # obses_list = tf.convert_to_tensor(obses_list)
        # obses_list = self.cal_prediction_obs(obs_real, mask_real, future_n_point_real)
        # pred_xs, pred_ys = obses_list[:, 3], obses_list[:, 4]

        self.render(path_values, path_index,  weights)
        self.recorder.record(obs_real, safe_action, self.step_timer.mean, path_index, path_values, self.ss_timer.mean, is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def render(self, path_values, path_index, weights, pred_xs=None, pred_ys=None):
        extension = 40
        dotted_line_style = '--'
        solid_line_style = '-'

        if not self.fig_plot:
            self.fig = plt.figure(figsize=(8, 8))
            self.fig_plot = 1
        plt.ion()

        plt.cla()
        ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
        for ax in self.fig.get_axes():
            ax.axis('off')
        patches = []
        ax.axis("equal")

        # ----------arrow--------------
        # plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 3, color='darkviolet')
        # plt.arrow(lane_width / 2, -square_length / 2 - 10 + 3, -0.5, 1.0, color='darkviolet', head_width=0.7)
        # plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='darkviolet', head_width=0.7)
        # plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 3, color='darkviolet')
        # plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 3, 0.5, 1.0, color='darkviolet', head_width=0.7)

        ax.add_patch(
            plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R), extension, Para.R_GREEN, edgecolor='white',
                          facecolor='green',
                          linewidth=1, alpha=0.7))
        ax.add_patch(
            plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L), extension, Para.L_GREEN,
                          edgecolor='white', facecolor='green',
                          linewidth=1, alpha=0.7))
        ax.add_patch(plt.Rectangle((Para.OFFSET_D_X - extension * math.cos(Para.ANGLE_D / 180 * pi),
                                    Para.OFFSET_D_Y - extension * math.sin(Para.ANGLE_D / 180 * pi)),
                                   Para.D_GREEN, extension, edgecolor='white', facecolor='green',
                                   angle=-(90 - Para.ANGLE_D), linewidth=1, alpha=0.7))

        # Left out lane
        for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
            lane_width_flag = [Para.L_OUT_0, Para.L_OUT_1, Para.L_OUT_2,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base = Para.OFFSET_L + Para.L_GREEN
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                     [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth)
        # Left in lane
        for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
            lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base = Para.OFFSET_L
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                     [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Right out lane
        for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
            lane_width_flag = [Para.R_OUT_0, Para.R_OUT_1, Para.R_OUT_2,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base = Para.OFFSET_R
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                     [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Right in lane
        for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
            lane_width_flag = [Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base = Para.OFFSET_R + Para.R_GREEN
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                     [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Up in lane
        for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
            lane_width_flag = [Para.U_IN_0, Para.U_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base_x, base_y = Para.OFFSET_U_X, Para.OFFSET_U_Y
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
            plt.plot([base_x - sum(lane_width_flag[:i]) * math.cos(
                (90 - Para.ANGLE_U) / 180 * pi) + extension * math.cos(
                Para.ANGLE_U / 180 * pi),
                      base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi)],
                     [base_y + sum(lane_width_flag[:i]) * math.sin(
                         (90 - Para.ANGLE_U) / 180 * pi) + extension * math.sin(
                         Para.ANGLE_U / 180 * pi),
                      base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Up out lane
        for i in range(0, Para.LANE_NUMBER_LON_OUT + 2):
            lane_width_flag = [Para.U_OUT_0, Para.U_OUT_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base_x, base_y = Para.OFFSET_U_X, Para.OFFSET_U_Y
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
            if i == 0:
                linestyle = solid_line_style
            plt.plot([base_x + sum(lane_width_flag[:i]) * math.cos(
                (90 - Para.ANGLE_U) / 180 * pi) + extension * math.cos(
                Para.ANGLE_U / 180 * pi),
                      base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi)],
                     [base_y - sum(lane_width_flag[:i]) * math.sin(
                         (90 - Para.ANGLE_U) / 180 * pi) + extension * math.sin(
                         Para.ANGLE_U / 180 * pi),
                      base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Down in lane
        for i in range(0, Para.LANE_NUMBER_LON_IN + 2):
            lane_width_flag = [Para.D_IN_0, Para.D_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base_x, base_y = Para.OFFSET_D_X + Para.D_GREEN * math.cos(
                (90 - Para.ANGLE_D) / 180 * pi), Para.OFFSET_D_Y - Para.D_GREEN * math.sin(
                (90 - Para.ANGLE_D) / 180 * pi)
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
            plt.plot([base_x + sum(lane_width_flag[:i]) * math.cos(
                (90 - Para.ANGLE_D) / 180 * pi) - extension * math.cos(
                Para.ANGLE_D / 180 * pi),
                      base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi)],
                     [base_y - sum(lane_width_flag[:i]) * math.sin(
                         (90 - Para.ANGLE_D) / 180 * pi) - extension * math.sin(
                         Para.ANGLE_D / 180 * pi),
                      base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # Down out lane
        for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
            lane_width_flag = [Para.D_OUT_0, Para.D_OUT_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base_x, base_y = Para.OFFSET_D_X, Para.OFFSET_D_Y
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
            plt.plot([base_x - sum(lane_width_flag[:i]) * math.cos(
                (90 - Para.ANGLE_D) / 180 * pi) - extension * math.cos(
                Para.ANGLE_D / 180 * pi),
                      base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi)],
                     [base_y + sum(lane_width_flag[:i]) * math.sin(
                         (90 - Para.ANGLE_D) / 180 * pi) - extension * math.sin(
                         Para.ANGLE_D / 180 * pi),
                      base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi)],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # roadblock
        roadblock_left = Wedge((Para.LEFT_X, Para.LEFT_Y), Para.ROADBLOCK_RADIUS, -90, 90, color='green', alpha=0.7)
        ax.add_patch(roadblock_left)
        roadblock_right = Wedge((Para.RIGHT_X, Para.RIGHT_Y), Para.ROADBLOCK_RADIUS, 90, -90, color='green', alpha=0.7)
        ax.add_patch(roadblock_right)

        # Oblique
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_U_X - (
                Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
            (90 - Para.ANGLE_U) / 180 * pi)],
                 [
                     Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                     Para.OFFSET_U_Y + (
                             Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                         (90 - Para.ANGLE_U) / 180 * pi)],
                 color='black', linewidth=1)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_D_X - (
                Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
            (90 - Para.ANGLE_D) / 180 * pi)],
                 [
                     Para.OFFSET_L - Para.L_IN_0 - Para.L_IN_1 - Para.L_IN_2 - Para.L_IN_3 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                     Para.OFFSET_D_Y + (
                             Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                         (90 - Para.ANGLE_D) / 180 * pi)],
                 color='black', linewidth=1)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_D_X + (
                          Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
                      (90 - Para.ANGLE_D) / 180 * pi)],
                 [Para.OFFSET_R - (
                         Para.R_OUT_0 + Para.R_OUT_1 + Para.R_OUT_2) - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                  Para.OFFSET_D_Y - (
                          Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                      (90 - Para.ANGLE_D) / 180 * pi)],
                 color='black', linewidth=1)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                  Para.OFFSET_U_X + (
                          Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
                      (90 - Para.ANGLE_U) / 180 * pi)],
                 [Para.OFFSET_R + (
                         Para.R_GREEN + Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3) + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                  Para.OFFSET_U_Y - (
                          Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                      (90 - Para.ANGLE_U) / 180 * pi)],
                 color='black', linewidth=1)

        # mark line
        line = [5, 10, 15, 20, 25, 30, 35, 40]
        color = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        for i in range(len(line)):
            plt.plot([Road.D_X1_U - line[i] * cos(Para.ANGLE_D * math.pi / 180),
                      Road.D_X3_U - line[i] * cos(Para.ANGLE_D * math.pi / 180)],
                     [Road.D_Y1_U - line[i] * sin(Para.ANGLE_D * math.pi / 180),
                      Road.D_Y3_U - line[i] * sin(Para.ANGLE_D * math.pi / 180)],
                     color=color[i], linewidth=1, alpha=0.3)
            plt.text(Road.D_X1_U - line[i] * cos(Para.ANGLE_D * math.pi / 180) - 3.5,
                     Road.D_Y1_U - line[i] * sin(Para.ANGLE_D * math.pi / 180), str(line[i]), fontsize=8)

        # stop line
        light_line_width = 2
        v_color_1, v_color_2, h_color_1, h_color_2 = 'gray', 'gray', 'gray', 'gray'
        lane_width_flag = [Para.D_IN_0, Para.D_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
        plt.plot([Para.OFFSET_D_X + Para.D_GREEN * math.cos((Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 [Para.OFFSET_D_Y + Para.D_GREEN * math.sin((Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 color=v_color_1, linewidth=light_line_width)
        plt.plot([Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
            (Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.cos(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 [Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                     (Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.sin(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 color='gray', linewidth=light_line_width)

        lane_width_flag = [Para.U_IN_0, Para.U_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
        plt.plot([Para.OFFSET_U_X,
                  Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
                 [Para.OFFSET_U_Y,
                  Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
                 color=v_color_1, linewidth=light_line_width)
        plt.plot([Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180),
                  Para.OFFSET_U_X + sum(lane_width_flag[:2]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
                 [Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180),
                  Para.OFFSET_U_Y + sum(lane_width_flag[:2]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
                 color='gray', linewidth=light_line_width)

        lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # left
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
                 color=h_color_1, linewidth=light_line_width)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
                 color=h_color_2, linewidth=light_line_width)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:3]), Para.OFFSET_L - sum(lane_width_flag[:4])],
                 color='gray', linewidth=light_line_width)

        lane_width_flag = [Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # right
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN,
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1])],
                 color=h_color_1, linewidth=light_line_width)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1]),
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3])],
                 color=h_color_2, linewidth=light_line_width)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3]),
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:4])],
                 color='gray', linewidth=light_line_width)

        # traffic light
        v_light = self.env.light_phase
        light_line_width = 2
        # 1 : left 2: straight
        if v_light == 0 or v_light == 1:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'green', 'green', 'red', 'red'
        elif v_light == 2:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'orange', 'orange', 'red', 'red'
        elif v_light == 3:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'
        elif v_light == 4:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'green'
        elif v_light == 5:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'orange'
        elif v_light == 6:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'
        elif v_light == 7:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'green', 'red'
        elif v_light == 8:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'orange', 'red'
        else:
            v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'red'

        lane_width_flag = [Para.D_IN_0, Para.D_IN_1,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
        plt.plot([Para.OFFSET_D_X + Para.D_GREEN * math.cos((Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 [Para.OFFSET_D_Y + Para.D_GREEN * math.sin((Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 color=v_color_1, linewidth=light_line_width)
        plt.plot([Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
            (Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.cos(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 [Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                     (Para.ANGLE_D - 90) * math.pi / 180),
                  Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.sin(
                      (Para.ANGLE_D - 90) * math.pi / 180)],
                 color='green', linewidth=light_line_width)

        lane_width_flag = [Para.U_IN_0, Para.U_IN_1,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
        plt.plot([Para.OFFSET_U_X,
                  Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
                 [Para.OFFSET_U_Y,
                  Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
                 color=v_color_1, linewidth=light_line_width)
        plt.plot([Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180),
                  Para.OFFSET_U_X + sum(lane_width_flag[:2]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
                 [Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180),
                  Para.OFFSET_U_Y + sum(lane_width_flag[:2]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
                 color='green', linewidth=light_line_width)

        lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # left
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
                 color=h_color_1, linewidth=light_line_width)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
                 color=h_color_2, linewidth=light_line_width)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_L - sum(lane_width_flag[:3]), Para.OFFSET_L - sum(lane_width_flag[:4])],
                 color='green', linewidth=light_line_width)

        lane_width_flag = [Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # right
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN,
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1])],
                 color=h_color_1, linewidth=light_line_width)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1]),
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3])],
                 color=h_color_2, linewidth=light_line_width)
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                 [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3]),
                  Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:4])],
                 color='green', linewidth=light_line_width)


        def is_in_plot_area(x, y, tolerance=5):
            if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                    -(Para.OFFSET_U_Y - Para.OFFSET_D_Y) / 2 - extension + tolerance < y < (Para.OFFSET_U_Y - Para.OFFSET_D_Y) / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(type, x, y, a, l, w, color, linestyle='-', patch=False):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            if patch:
                if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    item_color = 'purple'
                elif type == 'DEFAULT_PEDTYPE':
                    item_color = 'lime'
                else:
                    item_color = 'lightgray'
                patches.append(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color, facecolor=item_color,
                                             angle=-(90 - a), zorder=30))
            else:
                patches.append(matplotlib.patches.Rectangle(np.array([-l / 2 + x, -w / 2 + y]),
                                                            width=l, height=w,
                                                            fill=False,
                                                            facecolor=None,
                                                            edgecolor=color,
                                                            linestyle=linestyle,
                                                            linewidth=1.0,
                                                            transform=Affine2D().rotate_deg_around(*(x, y),
                                                                                                   a)))

        def draw_rotate_batch_rec(x, y, a, l, w):
            for i in range(len(x)):
                patches.append(matplotlib.patches.Rectangle(np.array([-l[i] / 2 + x[i], -w[i] / 2 + y[i]]),
                                                            width=l[i], height=w[i],
                                                            fill=False,
                                                            facecolor=None,
                                                            edgecolor='k',
                                                            linewidth=1.0,
                                                            transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                                                   a[i])))

        def plot_phi_line(type, x, y, phi, color):
            if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                line_length = 2
            elif type == 'DEFAULT_PEDTYPE':
                line_length = 1
            else:
                line_length = 5
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        # plot others
        filted_all_other = [item for item in self.env.all_other if is_in_plot_area(item['x'], item['y'])]
        other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
        other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
        other_as = np.array([item['phi'] for item in filted_all_other], np.float32)
        other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
        other_ws = np.array([item['w'] for item in filted_all_other], np.float32)

        draw_rotate_batch_rec(other_xs, other_ys, other_as, other_ls, other_ws)

        # plot interested others
        if weights is not None:
            assert weights.shape == (self.args.other_number,), print(weights.shape)
        for i in range(len(self.env.interested_other)):
            item = self.env.interested_other[i]
            item_mask = item['exist']
            item_x = item['x']
            item_y = item['y']
            item_phi = item['phi']
            item_l = item['l']
            item_w = item['w']
            item_type = item['type']
            if is_in_plot_area(item_x, item_y) and (item_mask == 1.0):
                plot_phi_line(item_type, item_x, item_y, item_phi, 'lightgray')
                draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='m', linestyle=':', patch=True)
                plt.text(item_x + 0.3, item_y + 2.5, str(round(item['v'] * 3.6, 2)))      # km/h
                # if (weights is not None) and (item_mask == 1.0):
                #     plt.text(item_x + 0.05, item_y + 0.15, "{:.2f}".format(weights[i]), color='purple', fontsize=12)

        # plot own car
        abso_obs = self.env._convert_to_abso(self.obs)
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac,obs_other = self.env._split_all(abso_obs)
        ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = obs_ego
        devi_lateral, devi_phi, devi_v = obs_track

        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'red')
        draw_rotate_rec('self_car', ego_x, ego_y, ego_phi, self.env.ego_l, self.env.ego_w, 'red')
        if (pred_xs is not None) and (pred_ys is not None):
            plt.plot(pred_xs, pred_ys, 'ro')

        # plot real time traj
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, item in enumerate(self.path_list):
            if i == path_index:
                plt.plot(item.path[0], item.path[1], color=color[i], alpha=1.0)
            else:
                plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
        _, point = self.env.ref_path._find_closest_point(ego_x, ego_y)
        path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]

        # text
        text_x, text_y_start = -110, 60
        ge = iter(range(0, 1000, 4))
        plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
        plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
        plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
        plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
        plt.text(text_x, text_y_start - next(ge), 'devi_lateral: {:.2f}m'.format(devi_lateral))
        plt.text(text_x, text_y_start - next(ge), 'devi_v: {:.2f}m/s'.format(devi_v))
        plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
        plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
        plt.text(text_x, text_y_start - next(ge), r'devi_phi: ${:.2f}\degree$'.format(devi_phi))

        plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
        plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 80, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        text_x, text_y_start = -60, 180
        plt.text(text_x, text_y_start - next(ge), 'Traffic mode:{}'.format(self.env.traffic_case), fontsize='large', bbox=dict(facecolor='red', alpha=0.5))

        # indicator for trajectory selection
        text_x, text_y_start = 25, -30
        ge = iter(range(0, 1000, 6))
        if path_values is not None:
            for i, value in enumerate(path_values):
                if i == path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
                             color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
                             color=color[i], fontstyle='italic')

        ax.add_collection(PatchCollection(patches, match_original=True))
        plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.jpg'.format(self.step_counter))


def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir, True)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('experiment-2021-12-11-21-35-50', 135000, logdir)  # for only path track

    for i in range(300):
        for _ in range(300):
            done = hier_decision.step()
            if done:
                print(hier_decision.env.done_type)
                break
        hier_decision.reset()


def plot_static_path():
    extension = 20
    dotted_line_style = '--'
    solid_line_style = '-'

    # plt.cla()
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")
    patches = []

    # ----------arrow--------------
    # plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 3, color='darkviolet')
    # plt.arrow(lane_width / 2, -square_length / 2 - 10 + 3, -0.5, 1.0, color='darkviolet', head_width=0.7)
    # plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='darkviolet', head_width=0.7)
    # plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 3, color='darkviolet')
    # plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 3, 0.5, 1.0, color='darkviolet', head_width=0.7)

    ax.add_patch(
        plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R), extension, Para.R_GREEN, edgecolor='white',
                      facecolor='green',
                      linewidth=1, alpha=0.7))
    ax.add_patch(
        plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L), extension, Para.L_GREEN,
                      edgecolor='white', facecolor='green',
                      linewidth=1, alpha=0.7))
    ax.add_patch(plt.Rectangle((Para.OFFSET_D_X - extension * math.cos(Para.ANGLE_D / 180 * pi),
                                Para.OFFSET_D_Y - extension * math.sin(Para.ANGLE_D / 180 * pi)),
                               Para.D_GREEN, extension, edgecolor='white', facecolor='green',
                               angle=-(90 - Para.ANGLE_D), linewidth=1, alpha=0.7))

    # Left out lane
    for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
        lane_width_flag = [Para.L_OUT_0, Para.L_OUT_1, Para.L_OUT_2,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base = Para.OFFSET_L + Para.L_GREEN
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                 [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                 linestyle=linestyle, color='black', linewidth=linewidth)
    # Left in lane
    for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
        lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base = Para.OFFSET_L
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                 [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Right out lane
    for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
        lane_width_flag = [Para.R_OUT_0, Para.R_OUT_1, Para.R_OUT_2,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base = Para.OFFSET_R
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                 [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Right in lane
    for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
        lane_width_flag = [Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3,
                           Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base = Para.OFFSET_R + Para.R_GREEN
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
        plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                 [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Up in lane
    for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
        lane_width_flag = [Para.U_IN_0, Para.U_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base_x, base_y = Para.OFFSET_U_X, Para.OFFSET_U_Y
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
        plt.plot([base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi) + extension * math.cos(
            Para.ANGLE_U / 180 * pi),
                  base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi)],
                 [base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi) + extension * math.sin(
                     Para.ANGLE_U / 180 * pi),
                  base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi)],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Up out lane
    for i in range(0, Para.LANE_NUMBER_LON_OUT + 2):
        lane_width_flag = [Para.U_OUT_0, Para.U_OUT_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base_x, base_y = Para.OFFSET_U_X, Para.OFFSET_U_Y
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
        if i == 0:
            linestyle = solid_line_style
        plt.plot([base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi) + extension * math.cos(
            Para.ANGLE_U / 180 * pi),
                  base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_U) / 180 * pi)],
                 [base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi) + extension * math.sin(
                     Para.ANGLE_U / 180 * pi),
                  base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_U) / 180 * pi)],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Down in lane
    for i in range(0, Para.LANE_NUMBER_LON_IN + 2):
        lane_width_flag = [Para.D_IN_0, Para.D_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base_x, base_y = Para.OFFSET_D_X + Para.D_GREEN * math.cos(
            (90 - Para.ANGLE_D) / 180 * pi), Para.OFFSET_D_Y - Para.D_GREEN * math.sin(
            (90 - Para.ANGLE_D) / 180 * pi)
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
        plt.plot([base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi) - extension * math.cos(
            Para.ANGLE_D / 180 * pi),
                  base_x + sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi)],
                 [base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi) - extension * math.sin(
                     Para.ANGLE_D / 180 * pi),
                  base_y - sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi)],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Down out lane
    for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
        lane_width_flag = [Para.D_OUT_0, Para.D_OUT_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
        base_x, base_y = Para.OFFSET_D_X, Para.OFFSET_D_Y
        linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
        linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
        plt.plot([base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi) - extension * math.cos(
            Para.ANGLE_D / 180 * pi),
                  base_x - sum(lane_width_flag[:i]) * math.cos((90 - Para.ANGLE_D) / 180 * pi)],
                 [base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi) - extension * math.sin(
                     Para.ANGLE_D / 180 * pi),
                  base_y + sum(lane_width_flag[:i]) * math.sin((90 - Para.ANGLE_D) / 180 * pi)],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # Oblique
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_U_X - (
                Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
        (90 - Para.ANGLE_U) / 180 * pi)],
             [
                 Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                 Para.OFFSET_U_Y + (
                             Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                     (90 - Para.ANGLE_U) / 180 * pi)],
             color='black', linewidth=1)
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_D_X - (
                Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
        (90 - Para.ANGLE_D) / 180 * pi)],
             [Para.OFFSET_L - Para.L_IN_0 - Para.L_IN_1 - Para.L_IN_2 - Para.L_IN_3 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
              Para.OFFSET_D_Y + (
                          Para.D_OUT_0 + Para.D_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                  (90 - Para.ANGLE_D) / 180 * pi)],
             color='black', linewidth=1)
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
              Para.OFFSET_D_X + (
                          Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
                  (90 - Para.ANGLE_D) / 180 * pi)],
             [Para.OFFSET_R - (
                         Para.R_OUT_0 + Para.R_OUT_1 + Para.R_OUT_2) - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
              Para.OFFSET_D_Y - (
                          Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                  (90 - Para.ANGLE_D) / 180 * pi)],
             color='black', linewidth=1)
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
              Para.OFFSET_U_X + (
                          Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
                  (90 - Para.ANGLE_U) / 180 * pi)],
             [Para.OFFSET_R + (Para.R_GREEN + Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3) + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
              Para.OFFSET_U_Y - (
                          Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                  (90 - Para.ANGLE_U) / 180 * pi)],
             color='black', linewidth=1)

    # stop line  # todo
    light_line_width = 2
    v_color_1, v_color_2, h_color_1, h_color_2 = 'gray', 'gray', 'gray', 'gray'
    lane_width_flag = [Para.D_IN_0, Para.D_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]     # Down
    plt.plot([Para.OFFSET_D_X + Para.D_GREEN * math.cos((Para.ANGLE_D - 90) * math.pi / 180),
              Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
                  (Para.ANGLE_D - 90) * math.pi / 180)],
             [Para.OFFSET_D_Y + Para.D_GREEN * math.sin((Para.ANGLE_D - 90) * math.pi / 180),
              Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                  (Para.ANGLE_D - 90) * math.pi / 180)],
             color=v_color_1, linewidth=light_line_width)
    plt.plot([Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.cos(
        (Para.ANGLE_D - 90) * math.pi / 180),
              Para.OFFSET_D_X + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.cos(
                  (Para.ANGLE_D - 90) * math.pi / 180)],
             [Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:1])) * math.sin(
                 (Para.ANGLE_D - 90) * math.pi / 180),
              Para.OFFSET_D_Y + (Para.D_GREEN + sum(lane_width_flag[:2])) * math.sin(
                  (Para.ANGLE_D - 90) * math.pi / 180)],
             color='gray', linewidth=light_line_width)

    lane_width_flag = [Para.U_IN_0, Para.U_IN_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]     # Up
    plt.plot([Para.OFFSET_U_X,
              Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
             [Para.OFFSET_U_Y,
              Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
             color=v_color_1, linewidth=light_line_width)
    plt.plot([Para.OFFSET_U_X + sum(lane_width_flag[:1]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180),
              Para.OFFSET_U_X + sum(lane_width_flag[:2]) * math.cos((Para.ANGLE_U + 90) * math.pi / 180)],
             [Para.OFFSET_U_Y + sum(lane_width_flag[:1]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180),
              Para.OFFSET_U_Y + sum(lane_width_flag[:2]) * math.sin((Para.ANGLE_U + 90) * math.pi / 180)],
             color='gray', linewidth=light_line_width)

    lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                       Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # left
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
             color=h_color_1, linewidth=light_line_width)
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
             color=h_color_2, linewidth=light_line_width)
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_L - sum(lane_width_flag[:3]), Para.OFFSET_L - sum(lane_width_flag[:4])],
             color='gray', linewidth=light_line_width)


    lane_width_flag = [Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]    # right
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_R + Para.R_GREEN,
              Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1])],
             color=h_color_1, linewidth=light_line_width)
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:1]),
              Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3])],
             color=h_color_2, linewidth=light_line_width)
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:3]),
              Para.OFFSET_R + Para.R_GREEN + sum(lane_width_flag[:4])],
             color='gray', linewidth=light_line_width)

    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list
        control_points = path.control_points
        color = ['royalblue', 'orangered', 'teal', 'darkviolet']
        path_list_copy = path_list['green']
        for i, (path_x, path_y, _, _) in enumerate(path_list_copy):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                plt.scatter(point[0], point[1], color=color[i], s=20, alpha=0.5)
            plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i], alpha=0.3)
            plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i], alpha=0.3)
            plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i], alpha=0.3)

    plt.savefig('./multipath_planning.pdf')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    interval = file_num // (num-1)
    start = file_num % (num-1)
    # print(start, file_num, interval)
    selected = [start//2] + [start//2+interval*i for i in range(1, num-1)]
    # print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.jpg'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.jpg'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)


