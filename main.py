# -*- coding: utf-8 -*-
import os
import time
import shutil
import subprocess
import magic
import re
from scipy import ndimage
# import cv2
import glob

import numpy as np
from protobuf_to_dict import protobuf_to_dict
import idc_info_pb2 as pb

import matplotlib
matplotlib.rcParams[u'font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge


from math import cos, sin, pi
from env_build.dynamics_and_models import ReferencePath

from env_build.utils.load_policy import LoadPolicy
from env_build.endtoend_env_utils import *
from render_utils import  *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataReplay(object):
    def __init__(self, idc_planner_info_list, try_dir, model_dir, iter, replay_speed):
        self.try_dir = try_dir
        self.info_list = idc_planner_info_list
        self.Load_policy = LoadPolicy('./models/{}'.format(model_dir), iter)
        self.Net = self.Load_policy.Net
        self.replay_speed = replay_speed
        self.task = None
        self.traffic_light = None
        self.ref_path = None
        self.total_time = None
        self.info = []
        self.ego_info = []
        self.other_info = []
        self.obs_info = []
        self.path_info = []
        self.decision_info = []
        self.plot_dict = dict(normalized_acc=[],
                              normalized_front_wheel=[],
                              normalized_acc_clamp=[],
                              normalized_front_wheel_clamp=[],
                              acc=[],
                              steer=[],
                              acc_real=[],
                              steer_real=[],
                              v_x=[],
                              v_y=[],
                              r=[],
                              decision_time_ms=[],
                              path_value=None)

        self.key2label = dict(normalized_acc='Acceleration [$\mathrm {m/s^2}$]',
                              normalized_front_wheel='Front wheel angle [$\circ$]',
                              normalized_acc_clamp='Acceleration [$\mathrm {m/s^2}$]',
                              normalized_front_wheel_clamp='Front wheel angle [$\circ$]',
                              acc='Acceleration [$\mathrm {m/s^2}$]',
                              steer='Steering angle [$\circ$]',
                              acc_real='Acceleration [$\mathrm {m/s^2}$]',
                              steer_real='Steering angle [$\circ$]',
                              v_x='Speed [m/s]',
                              v_y='Speed [m/s]',
                              r='Yaw rate [rad/s]',
                              decision_time_ms='Decision time [ms]',
                              path_value='Path value')
        self.get_info()
        self.interested_info = get_list_of_participants_in_obs(self.info_list)

    def get_info(self):
        self.task = self.info_list[0]['task']
        self.traffic_light = self.info_list[0]['traffic_light']
        self.ref_path = ReferencePath(self.task, 'green')
        self.plot_dict['path_value'] = get_list_of_path_values(self.info_list)
        # print(get_list_of_path_values(self.info_list))
        self.total_time = len(self.info_list)
        sss = 1
        for i in range(len(self.info_list)):
            info_dict = self.info_list[i]
            # ego state
            ego_state_dict = info_dict['ego_state']
            for _ in ['v_x', 'v_y', 'r']:
                if _ not in ego_state_dict.keys():
                    ego_state_dict[_] = 0
                    info_dict['ego_state'][_] = 0
                    self.info_list[i]['ego_state'][_] = 0
            self.ego_info.append(ego_state_dict)

            # other state
            try:
                self.other_info.append(info_dict['other_state'])
            except:
                self.other_info.append([])

            # decision info
            # add 0
            for _ in ['selected_path_idx', 'is_safe', 'normalized_front_wheel_clamp']:
                if _ not in info_dict['decision'].keys():
                    info_dict['decision'][_] = 0
                    self.info_list[i]['decision'][_] = 0

            # action bias
            selected_path_idx = info_dict['decision']['selected_path_idx']
            obses = np.array(info_dict['obs_vector'][selected_path_idx]['input_vector'], dtype=np.float32)
            mask = np.array(info_dict['mask_vector'][selected_path_idx]['input_vector'], dtype=np.float32)
            a_true, weights_true = self.Load_policy.run_batch(obses[np.newaxis, :], mask[np.newaxis, :])
            steer_bias = info_dict['decision']['normalized_front_wheel'] - a_true[0][0]
            acc_bias = info_dict['decision']['normalized_acc'] - a_true[0][1]

            # value bias
            value_bias = []
            for path_idx in range(len(info_dict['obs_vector'])):
                obses = np.array(info_dict['obs_vector'][path_idx]['input_vector'], dtype=np.float32)
                mask = np.array(info_dict['mask_vector'][path_idx]['input_vector'], dtype=np.float32)
                value_true = self.Load_policy.obj_value_batch(obses[np.newaxis, :], mask[np.newaxis, :])

                value_bias.append(info_dict['decision']['path_value'][path_idx] - value_true[0].numpy())

            # save the info to decision
            info_dict['decision']['steer_bias'] = steer_bias
            info_dict['decision']['acc_bias'] = acc_bias
            info_dict['decision']['value_bias'] = value_bias
            info_dict['decision']['weights_true'] = weights_true

            self.decision_info.append(info_dict['decision'])

            # plot_dict
            self.plot_dict['normalized_acc'].append(
                info_dict['decision']['normalized_acc'])
            self.plot_dict['normalized_front_wheel'].append(
                info_dict['decision']['normalized_front_wheel'] * STEER_SCALE * 180 / pi )
            self.plot_dict['normalized_acc_clamp'].append(
                info_dict['decision']['normalized_acc_clamp'])
            self.plot_dict['normalized_front_wheel_clamp'].append(
                info_dict['decision']['normalized_front_wheel_clamp'] * STEER_SCALE * 180 / pi)
            self.plot_dict['v_x'].append(
                info_dict['ego_state']['v_x'])
            self.plot_dict['v_y'].append(
                info_dict['ego_state']['v_y'])
            self.plot_dict['r'].append(
                info_dict['ego_state']['r'])
            self.plot_dict['decision_time_ms'].append(
                info_dict['decision']['decision_time_ns'] / 10e6)
            self.plot_dict['acc'].append(
                info_dict['decision']['normalized_acc_clamp'] * ACC_SCALE - ACC_SHIFT)
            self.plot_dict['steer'].append(
                info_dict['decision']['normalized_front_wheel_clamp'] * STEER_SCALE * STEER_RATIO * 180 / pi)
            # print(info_dict['traj_pose'][-1])
            self.plot_dict['acc_real'].append(
                info_dict['traj_pose'][-1]['y'] if 'y' in info_dict['traj_pose'][-1].keys() else 0)
            self.plot_dict['steer_real'].append(
                info_dict['traj_pose'][-1]['x'] if 'x' in info_dict['traj_pose'][-1].keys() else 0)

            # obs info
            obs_info_dict = {}
            obs_dim_list = [Para.EGO_ENCODING_DIM, Para.TRACK_ENCODING_DIM, Para.LIGHT_ENCODING_DIM,
                            Para.TASK_ENCODING_DIM, Para.REF_ENCODING_DIM, Para.HIS_ACT_ENCODING_DIM]

            selected_path_idx = info_dict['decision']['selected_path_idx']
            obs_info_dict['ego_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][:obs_dim_list[0]]
            obs_info_dict['track_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][obs_dim_list[0]:sum(obs_dim_list[:2])]
            obs_info_dict['light_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][sum(obs_dim_list[:2]):sum(obs_dim_list[:3])]
            obs_info_dict['task_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][sum(obs_dim_list[:3]):sum(obs_dim_list[:4])]
            obs_info_dict['ref_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][sum(obs_dim_list[:4]):sum(obs_dim_list[:5])]
            obs_info_dict['his_act_info'] = info_dict['obs_vector'][selected_path_idx]['input_vector'][sum(obs_dim_list[:5]):sum(obs_dim_list[:6])]
            self.obs_info.append(obs_info_dict)

            # process info
            processed_info_dict = {}
            processed_info_dict['traj_pose'] = info_dict['traj_pose']
            processed_info_dict['attn_vector'] = info_dict['attn_vector'][0]['input_vector']
            self.info.append(processed_info_dict)

    def replay(self, save_video=False):
        plt.ion()
        self.steer_img = plt.imread('steer.png') # TODO

        result_path = os.path.join(self.try_dir, 'replay_results')
        fig_path = os.path.join(self.try_dir, 'replay_results', 'figs')
        if save_video:
            if 'replay_results' in os.listdir(self.try_dir):
                shutil.rmtree(result_path)
                os.makedirs(fig_path)
            else:
                os.makedirs(fig_path)

        for i in range(0, self.total_time, self.replay_speed):
            self.ref_path.set_path(self.traffic_light, self.decision_info[i]['selected_path_idx'])
            self.plot_for_replay(self.ego_info[i], self.other_info[i], self.decision_info[i],
                                 save_video, replay_counter=i)
        if save_video:
            fig_path = os.path.join(self.try_dir, 'replay_results', 'figs')
            subprocess.call(['ffmpeg', '-framerate', '10', '-i', fig_path + '/' + '%03d.jpg',
                             self.try_dir + '/replay_results' + '/video.mp4'])
            # shutil.rmtree(fig_path)

    def plot_for_replay(self, ego_info, other_info, decision_info, save_video, replay_counter):
        extension = 40
        dotted_line_style = '--'
        solid_line_style = '-'
        font = {'family': 'serif',
                'style': 'italic',
                'weight': 'normal',
                'color': 'black',
                'size': 10
                }

        if save_video:
            # f = plt.figure(figsize=(100, 50))
            plt.clf()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
        else:
            plt.clf()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")

        def is_in_plot_area(x, y, tolerance=5):
            if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                    -(Para.OFFSET_U_Y - Para.OFFSET_D_Y) / 2 - extension + tolerance < y < (
                    Para.OFFSET_U_Y - Para.OFFSET_D_Y) / 2 + extension - tolerance:
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

        def draw_rotate_batch_rec(x, y, a, l, w, patch=False):
            if patch:
                for i in range(len(x)):
                    patches.append(matplotlib.patches.Rectangle(np.array([-l[i] / 2 + x[i], -w[i] / 2 + y[i]]),
                                                                width=l[i], height=w[i],
                                                                fill=False,
                                                                facecolor='red',
                                                                edgecolor='m',
                                                                linestyle=':',
                                                                linewidth=1.0,
                                                                transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                                                       a[i])))
            else:
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

        # def get_partici_type_str(partici_type):
        #     if partici_type[0] == 1.:
        #         return 'bike'
        #     elif partici_type[1] == 1.:
        #         return 'person'
        #     elif partici_type[2] == 1.:
        #         return 'veh'

        patches = []

        # comment
        if 'comment.txt' in os.listdir(self.try_dir):
            with open(self.try_dir + '/' + 'comment.txt', "r") as f:
                data = f.read()
                ax.text(-120, -60, data, wrap=True, fontsize=14)

        # config
        with open(os.path.dirname(self.try_dir) + '/' + 'exp_config.txt', "r") as f:
            config = f.read()
            ax.text(20, 40, config, fontsize=14)

        ax.add_patch(
            plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R), extension, Para.R_GREEN, edgecolor='white',
                          facecolor='green',
                          linewidth=1, alpha=0.7))
        ax.add_patch(
            plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension + Para.BIAS_LEFT_LAT, Para.OFFSET_L), extension,
                          Para.L_GREEN,
                          edgecolor='white', facecolor='green',
                          linewidth=1, alpha=0.7))
        ax.add_patch(
            plt.Rectangle((Para.OFFSET_D_X - extension * math.cos(Para.ANGLE_D / 180 * pi),
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
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension + Para.BIAS_LEFT_LAT,
                      -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT],
                     [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                     linestyle=linestyle, color='black', linewidth=linewidth)
        # Left in lane
        for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
            lane_width_flag = [Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            base = Para.OFFSET_L
            linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
            linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension + Para.BIAS_LEFT_LAT,
                      -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT],
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
        roadblock_right = Wedge((Para.RIGHT_X, Para.RIGHT_Y), Para.ROADBLOCK_RADIUS, 90, -90, color='green',
                                alpha=0.7)
        ax.add_patch(roadblock_right)

        # Oblique
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, Para.OFFSET_U_X - (
                Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.cos(
            (90 - Para.ANGLE_U) / 180 * pi)],
                 [
                     Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                     Para.OFFSET_U_Y + (
                             Para.U_IN_0 + Para.U_IN_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) * math.sin(
                         (90 - Para.ANGLE_U) / 180 * pi)],
                 color='black', linewidth=1)
        plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, Para.OFFSET_D_X - (
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
        plt.plot(
            [-Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT],
            [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
            color=h_color_1, linewidth=light_line_width)
        plt.plot(
            [-Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT],
            [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
            color=h_color_2, linewidth=light_line_width)
        plt.plot(
            [-Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT, -Para.CROSSROAD_SIZE_LAT / 2 + Para.BIAS_LEFT_LAT],
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

        # plot others
        filted_all_other = [item for item in other_info if is_in_plot_area(item['x'], item['y'])]
        other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
        other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
        other_as = np.array([item['phi'] for item in filted_all_other], np.float32) * 180 / pi
        other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
        other_ws = np.array([item['w'] for item in filted_all_other], np.float32)
        other_types = [item['type'] for item in filted_all_other]

        for i in range(len(other_xs)):
            if SHOW_OTHER:
                ax.text(other_xs[i] - 40, other_ys[i] + 0,
                        'x:{:.2f} y:{:.2f} phi:{:.2f} type{}'.format(other_xs[i], other_ys[i], other_as[i], other_types[i]),
                        color='black')
            plot_phi_line('self_car', other_xs[i], other_ys[i], other_as[i], 'red')

        draw_rotate_batch_rec(other_xs, other_ys, other_as, other_ls, other_ws, patch=False)

        # plot own car
        real_ego_x = ego_info['x']
        real_ego_y = ego_info['y']
        real_ego_phi = ego_info['phi'] * 180 / pi
        real_ego_v_x = ego_info['v_x']
        real_ego_v_y = ego_info['v_y']
        real_ego_r = ego_info['r']

        ego_l = 4.8
        ego_w = 2

        plot_phi_line('self_car', real_ego_x, real_ego_y, real_ego_phi, 'red')
        ax.text(real_ego_x, real_ego_y + 3,
                'x:{:.2f} y:{:.2f} phi:{:.2f}'.format(real_ego_x, real_ego_y, real_ego_phi), color='red')
        draw_rotate_rec('self_car', real_ego_x, real_ego_y, real_ego_phi, ego_l, ego_w, 'red')

        # acc TODO:shift
        processed_acc = self.decision_info[replay_counter]['normalized_acc_clamp'] * ACC_SCALE - ACC_SHIFT
        ax.text(-36, 55, 'processed acc:{:.2f}'.format(processed_acc))
        if processed_acc > 0:
            # is_acc_positve = plt.Circle((-30, 50), 2, color='g', alpha=processed_acc / 1.5)
            is_acc_positve = plt.Circle((-30, 50), 2, color='g', alpha=min(1, processed_acc / 1.5))
        else:
            is_acc_positve = plt.Circle((-30, 50), 2, color='r', alpha=min(1, processed_acc / -3))
        ax.add_patch(is_acc_positve)

        # plot real time traj

        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, item in enumerate(self.ref_path.path_list['green']):
            if REF_ENCODING[i] == self.ref_path.ref_encoding:
                plt.plot(item[0], item[1], color=color[i], alpha=1.0)
            else:
                plt.plot(item[0], item[1], color=color[i], alpha=0.3)

        # safety_shield
        ax.text(-56, 55, 'safety shield')
        if decision_info['is_safe']:
            is_safe = plt.Circle((-50, 50), 2, color='g', alpha=0.5)
        else:
            is_safe = plt.Circle((-50, 50), 2, color='r', alpha=0.5)
        ax.add_patch(is_safe)

        # plot interested vehicles
        interested_vehs = self.interested_info[replay_counter]

        interested_xs = np.array([item['x'] + real_ego_x for item in interested_vehs], np.float32)
        interested_ys = np.array([item['y'] + real_ego_y for item in interested_vehs], np.float32)
        interested_as = np.array([item['phi'] for item in interested_vehs], np.float32)
        interested_ls = np.array([item['l'] for item in interested_vehs], np.float32)
        interested_ws = np.array([item['w'] for item in interested_vehs], np.float32)
        interested_vs = np.array([item['v'] for item in interested_vehs], np.float32)
        interested_types = np.array([item['type'] for item in interested_vehs], np.float32)

        draw_rotate_batch_rec(interested_xs, interested_ys, interested_as, interested_ls, interested_ws, patch=True)

        for num in range(len(interested_vehs)):
            if SHOW_INTERESTED:
                ax.text(interested_xs[num] + -4, interested_ys[num] + 3.15,
                        "x:{:.2f} y:{:.2f} v:{:.2f} phi:{:.2f}{}".format(interested_xs[num], interested_ys[num], interested_vs[num], interested_as[num], interested_types[num]),
                        color='purple',
                        fontsize=12)
            ax.text(interested_xs[num] + 0.05, interested_ys[num] + 0.15,
                     "{:.2f}".format(self.info[replay_counter]['attn_vector'][num]), color='purple', fontsize=12)

        # stop line
        x_left = [-30, -30]
        ax.plot()

        # steering wheel
        ax_steer = plt.axes([0.6, 0, 0.2, 0.2])
        ax_steer.axis('off')
        ax_steer.axis("equal")
        ax.text(43, -40,
                'steering angle: {:.2f}'.format(decision_info['normalized_front_wheel_clamp']
                                                * STEER_SCALE * STEER_RATIO * 180 / pi),
                fontdict=font) # TODO
        image_rotate = np.clip(ndimage.rotate(self.steer_img,
                                              decision_info['normalized_front_wheel_clamp'] * STEER_SCALE * STEER_RATIO * 180 / pi,
                                              reshape=False, cval=1.), 0.0, 1.0)
        ax_steer.imshow(image_rotate)

        ax.text(-120, 70, 'try_dir: {}'.format(self.try_dir[-47:]), fontdict=font)

        text_x, text_y_start = -110, 60
        ge = iter(range(0, 1000, 4))


        ax.text(text_x, text_y_start - next(ge), 'traffic_light: {}'.format(self.traffic_light), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'task: {}'.format(self.task), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'selected_path_idx: {}'.format(decision_info['selected_path_idx']),
                 fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(real_ego_v_x), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(real_ego_v_y), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(real_ego_r), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'normalized_acc: {:.2f}({:.2f})'.format(decision_info['normalized_acc'],
                                                                                decision_info['acc_bias']), fontdict=font)
        ax.text(text_x, text_y_start - next(ge),
                 'normalized_acc_clamp: {:.2f}'.format(self.decision_info[replay_counter]['normalized_acc_clamp']),
                 fontdict=font)

        ax.text(text_x, text_y_start - next(ge),
                 r'normalized_front_wheel: {:.2f}({:.2f})'.format(decision_info['normalized_front_wheel'],
                                                        decision_info['steer_bias']), fontdict=font)
        ax.text(text_x, text_y_start - next(ge),
                 'normalized_front_wheel_clamp: {:.2f}'.format(
                     self.decision_info[replay_counter]['normalized_front_wheel_clamp']),
                 fontdict=font)

        ax.text(text_x, text_y_start - next(ge), 'decision_time: {:.2f}ms'
                 .format(decision_info['decision_time_ns'] / 10e6), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'safety_shield_time: {:.2f}ms'
                 .format(decision_info['safety_shield_time_ns'] / 10e6), fontdict=font)

        for path_index in range(len(decision_info['path_value'])):
            ax.text(text_x, text_y_start - next(ge),
                     'path{}: {:.2f}({:.2f})'.format(path_index, decision_info['path_value'][path_index],
                                                     decision_info['value_bias'][path_index]),
                     fontdict=font)

        ax.text(text_x, text_y_start - next(ge), 'ego_info: {}'.
                 format([round(i, 2) for i in self.obs_info[replay_counter]['ego_info']]), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'track_info: {}'.
                 format([round(i, 2) for i in self.obs_info[replay_counter]['track_info']]), fontdict=font)
        ax.text(text_x, text_y_start - next(ge),
                'light_info: {}'.format([round(i, 2) for i in self.obs_info[replay_counter]['light_info']], fontdict=font))

        ax.text(text_x, text_y_start - next(ge), 'task_info: {}'.
                 format(self.obs_info[replay_counter]['task_info']), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'ref_info: {}'.
                 format(self.obs_info[replay_counter]['ref_info']), fontdict=font)
        ax.text(text_x, text_y_start - next(ge), 'his_act_info: {}'.
                 format([round(i, 2) for i in self.obs_info[replay_counter]['his_act_info']], fontdict=font))

        # plot
        ax.add_collection(PatchCollection(patches, match_original=True))

        if save_video:
            fig_path = self.try_dir + '/replay_results' + '/figs'
            plt.show()
            plt.pause(0.001)
            plt.savefig(fig_path + '/{:03d}.jpg'.format(int(replay_counter / self.replay_speed)), dpi=200)
        else:
            # plt.show()
            plt.pause(0.001)

    def plot_fig(self):
        if 'replay_results' not in os.listdir(self.try_dir):
            os.makedirs(self.try_dir + '/replay_results')
        time_line = np.array([0.1 * k for k in range(self.total_time)])
        for key, value in self.plot_dict.items():
            f = plt.figure(key, figsize=(10, 10))
            if key == 'path_value':
                color = ['red', 'green', 'skyblue', 'pink']
                for path_idx in range(len(value)):
                    plt.plot(time_line, value[path_idx], linewidth=2, color=color[path_idx], label='Path {}'.format(str(path_idx)))
                plt.legend(fontsize=20)
            elif key == 'acc':
                plt.plot(time_line, value, linewidth=2, color='indigo', label='acc')
                plt.plot(time_line, self.plot_dict['acc_real'], linewidth=2, color='skyblue', label='acc_real')
            elif key == 'steer':
                plt.plot(time_line, value, linewidth=2, color='indigo', label='steer')
                plt.plot(time_line, self.plot_dict['steer_real'], linewidth=2, color='red', label='steer_real')
            else:
                plt.plot(time_line, value, linewidth=2, color='indigo')

            if key == 'acc':
                plt.ylim([-3, 1.5])
            elif key == 'acc_real':
                plt.ylim([-3, 1.5])
            elif key == 'normalized_acc':
                plt.ylim([-3, 1.5])
            elif key == 'normalized_acc_clamp':
                plt.ylim([-3, 1.5])
            elif key == 'steer':
                plt.ylim([-240, 240])
            elif key == 'steer_real':
                plt.ylim([-240, 240])
            elif key == 'normalized_front_wheel':
                plt.ylim([-240, 240])
            elif key == 'normalized_front_wheel_clamp':
                plt.ylim([-240, 240])
            elif key == 'v_x':
                plt.ylim([-0.5, 6.])
            elif key == 'v_y':
                plt.ylim([-0.5, 0.5])
            elif key == 'r':
                plt.ylim([-0.4, 0.4])
            elif key == 'decision_time_ms':
                plt.ylim([-0.2, 10])
            else:
                assert key == 'path_value', 'oops, wrong key'
                plt.ylim([-0.2, 100])
            plt.xlabel("Time [s]", fontsize=20)
            plt.ylabel(self.key2label[key], fontsize=20)
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.savefig(self.try_dir + '/replay_results' + '/{}.pdf'.format(key))
            plt.close(f)


def get_replay_data(try_path, start_time=0):
    filepaths = os.listdir(try_path)
    idc_planner_info_list = []
    for filepath in filepaths:
        if is_binwary_file(try_path+'/'+filepath):
            binfile = open(try_path+'/'+filepath, 'rb')  # 打开二进制文件
            size = os.path.getsize(try_path+'/'+filepath)  # 获得文件大小
            data = binfile.read(size)
            binfile.close()
            IdcPlannerInfo = pb.IdcPlannerInfo()
            IdcPlannerInfo.ParseFromString(data)
            d = protobuf_to_dict(IdcPlannerInfo)
            idc_planner_info_list.append(d)
    idc_planner_info_list = sorted(idc_planner_info_list, key=lambda x: x['timestamp'])
    return idc_planner_info_list[start_time:]


def is_binwary_file(ff):
    mime_kw = 'x-executable|x-sharedlib|octet-stream|x-object'
    try:
        magic_mime = magic.from_file(ff, mime=True)
        magic_hit = re.search(mime_kw, magic_mime, re.I)
        if magic_hit:
            return True
        else:
            return False
    except Exception as e:
        return False


def image2video(forder):
    os.chdir(forder)
    subprocess.call(['ffmpeg', '-framerate', '10', '-i', 'step%03d.png', 'video.mp4'])


def get_list_of_participants_in_obs(input_):
    other_start_dim = 21
    max_other_num = 18

    def get_list_of_participants_dict(msg):
        selected_path_idx = msg['decision'].get('selected_path_idx', None)
        selected_path_idx = selected_path_idx if selected_path_idx else 0
        selected_obs = msg['obs_vector'][selected_path_idx]['input_vector']
        # attn_weights = msg[]
        out = []
        for i in range(max_other_num):
            other_vector = selected_obs[other_start_dim + i * 10:other_start_dim + (i + 1) * 10]
            parti_dict = dict(zip(['x', 'y', 'v', 'phi', 'l', 'w'], other_vector[:6]))
            parti_dict.setdefault('type', other_vector[6:9])
            out.append(parti_dict)
        return out
    return list(map(lambda msg: get_list_of_participants_dict(msg), input_))


def get_list_of_path_values(input_):
    path_num = len(input_[0]['decision']['path_value'])

    def find_the_ith_value(msg, i):
        return msg['decision']['path_value'][i]

    out = []
    for i in range(path_num):
        out.append(list(map(lambda msg: find_the_ith_value(msg, i), input_)))
    return out


def get_alpha(acc):
    return acc / 1.5 if acc > 0 else acc / -3


def main():
    # os.makedirs('~/test_input')
    try_path = '/home/tly/render4didi_proj/test/test_20211224_2pm/exp_2021_12_24_16_48_54/try_2021_12_24_16_58_49'
    model_dir = 'experiment-2021-12-16-00-54-59'
    iter = 300000
    replay_speed = 3
    replay_data = get_replay_data(try_path, start_time=20)
    data_replay = DataReplay(replay_data, try_path, model_dir=model_dir, iter=iter, replay_speed=replay_speed)
    data_replay.replay(save_video=True)
    data_replay.plot_fig()
    # for filepath in os.listdir(try_path):
    #     if is_binwary_file(try_path+'/'+filepath):
    #         os.remove(try_path+'/'+filepath)


def test():
    test_dir_ = 'test/test_20211224_2pm'
    test_dir = os.path.join(os.getcwd(), test_dir_)
    model_dir = 'experiment-2021-12-16-00-54-59'
    iter = 300000
    replay_speed = 3
    exps = [x for x in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, x))]
    for exp in sorted(exps, key=lambda x: int(x[15:17].strip('_')) * 60 + int(x[18:20].strip('_'))):
        exp_dir = os.path.join(test_dir, exp)
        exp_dir = '/home/tly/render4didi_proj/test/test_20211224_2pm/exp_2021_12_24_16_27_9'
        # print(exp_dir)
        trials = [x for x in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, x))]
        for trial in sorted(trials, key=lambda x: int(x[15:17].strip('_')) * 60 + int(x[18:20].strip('_'))):
            try_dir = os.path.join(exp_dir, trial)
            print(try_dir)
            replay_data = get_replay_data(try_dir)
            data_replay = DataReplay(replay_data, try_dir, model_dir=model_dir, iter=iter, replay_speed=replay_speed)
            data_replay.replay(save_video=True)
            data_replay.plot_fig()


if __name__ == '__main__':
    main()