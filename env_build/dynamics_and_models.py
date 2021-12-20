#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and
from matplotlib.patches import Wedge

from env_build.endtoend_env_utils import *

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# class VehicleDynamics(object):
#     def __init__(self, ):
#         # self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
#         #                            C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
#         #                            a=1.06,  # distance from CG to front axle [m]
#         #                            b=1.85,  # distance from CG to rear axle [m]
#         #                            mass=1412.,  # mass [kg]
#         #                            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
#         #                            miu=1.0,  # tire-road friction coefficient
#         #                            g=9.81,  # acceleration of gravity [m/s^2]
#         #                            )
#         self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
#                                    C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
#                                    a=1.19,  # distance from CG to front axle [m]
#                                    b=1.46,  # distance from CG to rear axle [m]
#                                    mass=1520.,  # mass [kg]
#                                    I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
#                                    miu=0.8,  # tire-road friction coefficient
#                                    g=9.81,  # acceleration of gravity [m/s^2]
#                                    )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
#         v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
#         phi = phi * np.pi / 180.
#         steer, a_x = actions[:, 0], actions[:, 1]
#         C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
#         C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
#         a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
#         b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
#         mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
#         I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
#         miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
#         g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)
#
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
#         F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
#         miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
#         miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
#         alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - steer
#         alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))
#
#         next_state = [v_x + tau * (a_x + v_y * r),
#                       (mass * v_y * v_x + tau * (
#                               a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
#                           v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
#                       (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
#                               tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
#                       x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
#                       y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
#                       (phi + tau * r) * 180 / np.pi]
#
#         return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)
#
#     def prediction(self, x_1, u_1, frequency):
#         x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
#         return x_next, next_params


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))

        next_v_x = v_x + tau * a_x
        next_x = x + tau * v_x * tf.cos(phi)
        next_y = y + tau * v_x * tf.sin(phi)
        next_phi = phi + tau * v_x * tf.tan(steer) / 2.865
        next_phi_deg = next_phi * 180 / np.pi
        next_r = v_x * tf.tan(steer) / 2.865
        next_v_y = tf.zeros_like(next_v_x)
        next_state = [next_v_x, next_v_y, next_r, next_x, next_y, next_phi_deg]
        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class EnvironmentModel(object):  # all tensors
    def __init__(self, mode='training'):
        self.mode = mode
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.actions = None
        self.reward_info = None
        self.future_n_points = None
        self.path_len = None
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.his_act_info_dim = Para.HIS_ACT_ENCODING_DIM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.light_info_dim,
                                    self.task_info_dim, self.ref_info_dim, self.his_act_info_dim])
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM

    def reset(self, obses, future_n_points):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None
        self.future_n_points = future_n_points
        self.path_len = self.future_n_points.shape[-1]

    def rollout_out(self, actions, step=0):  # ref_points [#batch, 4]
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                veh2person4real, veh2speed4real, self.reward_info = self.compute_rewards(self.obses, self.actions, actions, step)

            self.obses = self.compute_next_obses(self.obses, self.actions, actions)

        return self.obses, rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
               veh2bike4real, veh2person4real, veh2speed4real

    def rollout_out_online(self, actions):  # ref_points [#batch, 4]
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                veh2person4real, veh2speed4real, reward_dict = self.compute_rewards(self.obses, self.actions, actions)

            self.obses = self.compute_next_obses(self.obses, self.actions, actions)

        return self.obses, rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
               veh2bike4real, veh2person4real, veh2speed4real

    def compute_rewards(self, obses, actions, untransformed_action, step=0):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_bike, obses_person, obses_veh = self._split_other(obses_other)

        with tf.name_scope('compute_reward') as scope:
            bike_infos = tf.stop_gradient(obses_bike)
            person_infos = tf.stop_gradient(obses_person)
            veh_infos = tf.stop_gradient(obses_veh)

            steers, a_xs = actions[:, 0], actions[:, 1]
            steers_t_minus_2, a_x_t_minus_2 = obses_his_ac[:, 0], obses_his_ac[:, 1]
            steers_t_minus_1, a_x_t_minus_1 = obses_his_ac[:, 2], obses_his_ac[:, 3]
            steers_t, a_x_t = untransformed_action[:, 0], untransformed_action[:, 1]

            # rewards related to action and its derivatives
            steers_1st_order = (steers_t - steers_t_minus_1) * self.base_frequency
            steers_2nd_order = (steers_t - 2 * steers_t_minus_1 + steers_t_minus_2) * (
                    self.base_frequency ** 2)
            a_xs_1st_order = (a_x_t - a_x_t_minus_1) * self.base_frequency
            a_xs_2nd_order = (a_x_t - 2 * a_x_t_minus_1 + a_x_t_minus_2) * (
                    self.base_frequency ** 2)

            punish_steer0, punish_steer1, punish_steer2 = - tf.square(steers), - tf.square(steers_1st_order), - tf.square(steers_2nd_order)
            punish_a_x0, punish_a_x1, punish_a_x2 = - tf.square(a_xs), - tf.square(a_xs_1st_order), - tf.square(a_xs_2nd_order)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(obses_ego[:, 2])

            # rewards related to tracking error
            devi_lateral = -tf.square(obses_track[:, 0])
            devi_phi = -tf.cast(tf.square(obses_track[:, 1] * np.pi / 180.), dtype=tf.float32)
            devi_v = -tf.square(obses_track[:, 2])

            # rewards related to veh2veh collision
            ego_lws = (Para.L - Para.W) / 2.
            ego_front_points = tf.cast(obses_ego[:, 3] + ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(obses_ego[:, 4] + ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(obses_ego[:, 3] - ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(obses_ego[:, 4] - ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)

            delta_ego_vs = obses_track[:, 2]    # delta_ego_vs = ego_vs - ref_vs
            veh2speed4training = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))
            veh2speed4real = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))

            veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            for veh_index in range(self.veh_num):
                vehs = veh_infos[:, veh_index * self.per_other_info_dim:(veh_index + 1) * self.per_other_info_dim]
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        veh2veh4training += tf.where(veh2veh_dist - 3.5 < 0, tf.square(veh2veh_dist - 3.5),
                                                     tf.zeros_like(veh_infos[:, 0]))
                        veh2veh4real += tf.where(veh2veh_dist - 2.5 < 0, tf.square(veh2veh_dist - 2.5),
                                                 tf.zeros_like(veh_infos[:, 0]))

            veh2bike4real = tf.zeros_like(veh_infos[:, 0])
            veh2bike4training = tf.zeros_like(veh_infos[:, 0])
            for bike_index in range(self.bike_num):
                bikes = bike_infos[:, bike_index * self.per_other_info_dim:(bike_index + 1) * self.per_other_info_dim]
                bike_lws = (bikes[:, 4] - bikes[:, 5]) / 2.
                bike_front_points = tf.cast(bikes[:, 0] + bike_lws * tf.cos(bikes[:, 3] * np.pi / 180.),
                                            dtype=tf.float32), \
                                    tf.cast(bikes[:, 1] + bike_lws * tf.sin(bikes[:, 3] * np.pi / 180.),
                                            dtype=tf.float32)
                bike_rear_points = tf.cast(bikes[:, 0] - bike_lws * tf.cos(bikes[:, 3] * np.pi / 180.),
                                           dtype=tf.float32), \
                                   tf.cast(bikes[:, 1] - bike_lws * tf.sin(bikes[:, 3] * np.pi / 180.),
                                           dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for bike_point in [bike_front_points, bike_rear_points]:
                        veh2bike_dist = tf.sqrt(
                            tf.square(ego_point[0] - bike_point[0]) + tf.square(ego_point[1] - bike_point[1]))
                        veh2bike4training += tf.where(veh2bike_dist - 3.5 < 0, tf.square(veh2bike_dist - 3.5),
                                                      tf.zeros_like(veh_infos[:, 0]))
                        veh2bike4real += tf.where(veh2bike_dist - 2.5 < 0, tf.square(veh2bike_dist - 2.5),
                                                  tf.zeros_like(veh_infos[:, 0]))

            veh2person4real = tf.zeros_like(veh_infos[:, 0])
            veh2person4training = tf.zeros_like(veh_infos[:, 0])
            for person_index in range(self.person_num):
                persons = person_infos[:,
                          person_index * self.per_other_info_dim:(person_index + 1) * self.per_other_info_dim]
                person_point = tf.cast(persons[:, 0], dtype=tf.float32), tf.cast(persons[:, 1], dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2person_dist = tf.sqrt(
                        tf.square(ego_point[0] - person_point[0]) + tf.square(ego_point[1] - person_point[1]))
                    veh2person4training += tf.where(veh2person_dist - 3.5 < 0, tf.square(veh2person_dist - 4.5),
                                                    tf.zeros_like(veh_infos[:, 0]))
                    veh2person4real += tf.where(veh2person_dist - 2.5 < 0, tf.square(veh2person_dist - 2.5),
                                                tf.zeros_like(veh_infos[:, 0]))

            veh2road4real = tf.zeros_like(veh_infos[:, 0])
            veh2road4training = tf.zeros_like(veh_infos[:, 0])
            if step < 1:
                self.green_belt_flag = (obses_ego[:, 4] < Road.D_Y1_U)
            for ego_point in [ego_front_points, ego_rear_points]:
                left_flag = tf.reduce_all(tf.math.equal(obses_task, [[1., 0., 0.]]), axis=1, keepdims=False)
                straight_flag = tf.reduce_all(tf.math.equal(obses_task, [[0., 1., 0.]]), axis=1, keepdims=False)
                right_flag = tf.reduce_all(tf.math.equal(obses_task, [[0., 0., 1.]]), axis=1, keepdims=False)

                # dis>0: right; dis<0: left
                # start lane for left and straight
                dis_d1_left = (ego_point[0] * Road.D_K1 - ego_point[1] + Road.D_B1) / (tf.pow((tf.square(Road.D_K1)+1), 0.5))
                dis_d1_right = (ego_point[0] * Road.D_K2 - ego_point[1] + Road.D_B2) / (tf.pow((tf.square(Road.D_K2)+1), 0.5))

                # veh2road4training += tf.where(logical_and(left_flag, logical_and(ego_point[1] < Road.D_Y1_U + 1.5 * Para.L, dis_d1_left < 1.0)), tf.square(dis_d1_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += 1.5 * tf.where(logical_and(left_flag, logical_and(self.green_belt_flag, dis_d1_left < 1.0)), tf.square(dis_d1_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += 1.2 * tf.where(logical_and(left_flag, logical_and(self.green_belt_flag, dis_d1_right > -1.0)), tf.square(dis_d1_right - (-1.0)), tf.zeros_like(veh_infos[:, 0]))

                veh2road4training += 1.5 * tf.where(logical_and(straight_flag, logical_and(self.green_belt_flag, dis_d1_left < 1.0)), tf.square(dis_d1_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                # veh2road4training += tf.where(logical_and(straight_flag, logical_and(ego_point[1] < Road.D_Y1_U + 1.5 * Para.L, dis_d1_left < 1.0)), tf.square(dis_d1_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += 1.2 * tf.where(logical_and(straight_flag, logical_and(self.green_belt_flag, dis_d1_right > -1.0)), tf.square(dis_d1_right - (-1.0)), tf.zeros_like(veh_infos[:, 0]))

                # start lane for right
                dis_d2_left = (ego_point[0] * Road.D_K2 - ego_point[1] + Road.D_B2) / (tf.pow((tf.square(Road.D_K2)+1), 0.5))
                dis_d2_right = (ego_point[0] * Road.D_K3 - ego_point[1] + Road.D_B3) / (tf.pow((tf.square(Road.D_K3)+1), 0.5))

                veh2road4training += tf.where(logical_and(right_flag, logical_and(self.green_belt_flag, dis_d2_left < 1.0)), tf.square(dis_d2_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += 1.2 * tf.where(logical_and(right_flag, logical_and(self.green_belt_flag, dis_d2_right > -1.0)), tf.square(dis_d2_right - (-1.0)) , tf.zeros_like(veh_infos[:, 0]))

                # end line for straight
                dis_u_left = (ego_point[0] * Road.U_K1 - ego_point[1] + Road.U_B1) / (tf.pow((tf.square(Road.U_K1)+1), 0.5))
                dis_u_right = (ego_point[0] * Road.U_K2 - ego_point[1] + Road.U_B2) / (tf.pow((tf.square(Road.U_K2)+1), 0.5))

                veh2road4training += tf.where(logical_and(straight_flag, logical_and(ego_point[1] > Road.U_Y1_D - 1.5 * Para.L, dis_u_left < 1.0)), tf.square(dis_u_left - 1.0), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += tf.where(logical_and(straight_flag, logical_and(ego_point[1] > Road.U_Y2_D, dis_u_right > -1.0)), tf.square(dis_u_right  - (-1.0)), tf.zeros_like(veh_infos[:, 0]))

                # end line for left
                veh2road4training += tf.where(logical_and(left_flag,logical_and(ego_point[0] < -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH + Para.CROSSROAD_SIZE_LAT / 3, ego_point[1] > Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 - 1.0)),
                    tf.square(ego_point[1] - (Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 - 1.0)), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += tf.where(logical_and(left_flag,logical_and(ego_point[0] < -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH, ego_point[1] < Para.OFFSET_L + Para.L_GREEN + 1.0)),
                    tf.square(ego_point[1] - (Para.OFFSET_L + Para.L_GREEN + 1.0)), tf.zeros_like(veh_infos[:, 0]))

                # roadblock for left
                dis_left = tf.sqrt(tf.square(ego_point[0] - Para.LEFT_X) + tf.square(ego_point[1] - Para.LEFT_Y))
                veh2road4training += tf.where(logical_and(left_flag, dis_left < Para.ROADBLOCK_RADIUS + 1.0), tf.square(dis_left - (Para.ROADBLOCK_RADIUS + 1.0)), tf.zeros_like(veh_infos[:, 0]))

                # end line for right
                veh2road4training += tf.where(logical_and(right_flag, logical_and(ego_point[0] > Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH - Para.CROSSROAD_SIZE_LAT / 3, ego_point[1] > Para.OFFSET_R - 1.0)),
                    tf.square(ego_point[1] - (Para.OFFSET_R - 1.0)), tf.zeros_like(veh_infos[:, 0]))
                veh2road4training += tf.where(logical_and(right_flag, logical_and(ego_point[0] > Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH, ego_point[1] < Para.OFFSET_R - Para.L_OUT_0 - Para.L_OUT_1 - Para.L_OUT_2 + 1.0)),
                    tf.square(ego_point[1] - (Para.OFFSET_R - Para.L_OUT_0 - Para.L_OUT_1 - Para.L_OUT_2 + 1.0)), tf.zeros_like(veh_infos[:, 0]))

                # roadblock for right
                dis_right = tf.sqrt(tf.square(ego_point[0] - Para.RIGHT_X) + tf.square(ego_point[1] - Para.RIGHT_Y))
                veh2road4training += tf.where(logical_and(right_flag, dis_right < Para.ROADBLOCK_RADIUS + 1.0), tf.square(dis_right - (Para.ROADBLOCK_RADIUS + 1.0)), tf.zeros_like(veh_infos[:, 0]))

            veh2road4real = veh2road4training

            rewards = 0.15 * devi_v + 0.1 * devi_lateral + 10 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.0 * punish_steer0 + 0.4 * punish_steer1 + 0.1 * punish_steer2 + \
                      0.05 * punish_a_x0 + 0.0 * punish_a_x1 + 0.0 * punish_a_x2

            rewards4value = 0.15 * devi_v + 0.1 * devi_lateral + 10 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.0 * punish_steer0 + 0.05 * punish_a_x0

            punish_term_for_training = veh2veh4training + veh2road4training + veh2bike4training + veh2person4training + veh2speed4training
            real_punish_term = veh2veh4real + veh2road4real + veh2bike4real + veh2person4real + veh2speed4real

            reward_dict = dict(punish_steer0=punish_steer0,
                               punish_steer1=punish_steer1,
                               punish_steer2=punish_steer2,
                               punish_a_x0=punish_a_x0,
                               punish_a_x1=punish_a_x1,
                               punish_a_x2=punish_a_x2,
                               punish_yaw_rate=punish_yaw_rate,
                               devi_v=devi_v,
                               devi_lateral=devi_lateral,
                               devi_phi=devi_phi,
                               scaled_punish_steer0=0.0 * punish_steer0,
                               scaled_punish_steer1=0.4 * punish_steer1,
                               scaled_punish_steer2=0.1 * punish_steer2,
                               scaled_punish_a_x0=0.05 * punish_a_x0,
                               scaled_punish_a_x1=0.0 * punish_a_x1,
                               scaled_punish_a_x2=0.0 * punish_a_x2,
                               scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
                               scaled_devi_v=0.15 * devi_v,
                               scaled_devi_lateral=0.1 * devi_lateral,
                               scaled_devi_phi=10 * devi_phi,
                               veh2veh4training=veh2veh4training,
                               veh2road4training=veh2road4training,
                               veh2bike4training=veh2bike4training,
                               veh2person4training=veh2person4training,
                               veh2speedtraining=veh2speed4training,
                               veh2speedreal=veh2speed4real,
                               veh2veh4real=veh2veh4real,
                               veh2road4real=veh2road4real,
                               veh2bike2real=veh2bike4real,
                               veh2person2real=veh2person4real
                               )

            return rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                   veh2person4real, veh2speed4real, reward_dict

    def compute_next_obses(self, obses, actions, untransformed_actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_other = tf.stop_gradient(obses_other)
        next_obses_ego = self._ego_predict(obses_ego, actions)
        next_obses_track = self._compute_next_track_info_vectorized(next_obses_ego)
        next_obses_his_ac = tf.concat([obses_his_ac[:, -2:], untransformed_actions], axis=-1)
        next_obses_other = self._other_predict(obses_other)
        next_obses = tf.concat([next_obses_ego, next_obses_track, obses_light, obses_task, obses_ref, next_obses_his_ac, next_obses_other],
                               axis=-1)
        next_obses = self._convert_to_rela(next_obses)
        return next_obses

    def _find_closest_point_batch(self, xs, ys, paths):
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), [1, self.path_len])
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), [1, self.path_len])
        pathx_tile = paths[:, 0, :]
        pathy_tile = paths[:, 1, :]
        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)
        indexs = tf.argmin(dist_array, 1)
        ref_points = tf.gather(paths, indices=indexs, axis=-1, batch_dims=1)
        return indexs, ref_points

    def _compute_next_track_info_vectorized(self, next_ego_infos):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis = [next_ego_infos[:, i] for i in range(self.ego_info_dim)]

        # find close point
        indexes, ref_points = self._find_closest_point_batch(ego_xs, ego_ys, self.future_n_points)

        ref_xs, ref_ys, ref_phis, ref_vs = [ref_points[:, i] for i in range(4)]
        ref_phis_rad = ref_phis * np.pi / 180

        vector_ref_phi = tf.stack([tf.cos(ref_phis_rad), tf.sin(ref_phis_rad)], axis=-1)
        vector_ref_phi_ccw_90 = tf.stack([-tf.sin(ref_phis_rad), tf.cos(ref_phis_rad)], axis=-1) # ccw for counterclockwise
        vector_ego2ref = tf.stack([ref_xs - ego_xs, ref_ys - ego_ys], axis=-1)

        signed_dist_longi = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi, axis=-1))
        signed_dist_lateral = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi_ccw_90, axis=-1))

        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - ref_vs
        return tf.stack([signed_dist_lateral, delta_phi, delta_vs], axis=-1)

    def _convert_to_rela(self, obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat([tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
                        axis=-1)
        ego = tf.expand_dims(ego, 1)
        rela = obses_other_reshape - ego
        rela_obses_other = self._reshape_other(rela, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, rela_obses_other], axis=-1)

    def _convert_to_abso(self, rela_obses):
        obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(rela_obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat([tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
                        axis=-1)
        ego = tf.expand_dims(ego, 1)
        abso = obses_other_reshape + ego
        abso_obses_other = self._reshape_other(abso, reverse=True)

        return tf.concat([obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, abso_obses_other], axis=-1)

    def _ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                       ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis], axis=-1)
        return ego_next_infos

    def _other_predict(self, obses_other):
        obses_other_reshape = self._reshape_other(obses_other)

        xs, ys, vs, phis, turn_rad = obses_other_reshape[:, :, 0], obses_other_reshape[:, :, 1], \
                                     obses_other_reshape[:, :, 2], obses_other_reshape[:, :, 3], \
                                     obses_other_reshape[:, :, -1]
        phis_rad = phis * np.pi / 180.

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)
        phis_rad_delta = vs / self.base_frequency * turn_rad

        next_xs, next_ys, next_vs, next_phis_rad = xs + xs_delta, ys + ys_delta, vs, phis_rad + phis_rad_delta
        next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)
        next_phis = next_phis_rad * 180 / np.pi
        next_info = tf.concat([tf.stack([next_xs, next_ys, next_vs, next_phis], -1), obses_other_reshape[:, :, 4:]],
                              axis=-1)
        next_obses_other = self._reshape_other(next_info, reverse=True)
        return next_obses_other

    def _split_all(self, obses):
        obses_ego = obses[:, :self.ego_info_dim]
        obses_track = obses[:, self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        obses_light = obses[:, self.ego_info_dim + self.track_info_dim:
                               self.ego_info_dim + self.track_info_dim + self.light_info_dim]
        obses_task = obses[:, self.ego_info_dim + self.track_info_dim + self.light_info_dim:
                              self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim]
        obses_ref = obses[:, self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim:
                             self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim]
        obses_his_ac = obses[:, self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim:
                                self.other_start_dim]
        obses_other = obses[:, self.other_start_dim:]

        return obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other

    def _split_other(self, obses_other):
        obses_bike = obses_other[:, :self.bike_num * self.per_other_info_dim]
        obses_person = obses_other[:, self.bike_num * self.per_other_info_dim:
                                      (self.bike_num + self.person_num) * self.per_other_info_dim]
        obses_veh = obses_other[:, (self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obses_bike, obses_person, obses_veh

    def _reshape_other(self, obses_other, reverse=False):
        if reverse:
            return tf.reshape(obses_other, (-1, self.other_number * self.per_other_info_dim))
        else:
            return tf.reshape(obses_other, (-1, self.other_number, self.per_other_info_dim))

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.4 * steer_norm, 2.25 * a_xs_norm - 0.75
        return tf.stack([steer_scale, a_xs_scale], 1)

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            extension = 40
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axis("equal")

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
            # roadblock
            roadblock_left = Wedge((Para.LEFT_X, Para.LEFT_Y), Para.ROADBLOCK_RADIUS, -90, 90, color='green', alpha=0.7)
            ax.add_patch(roadblock_left)
            roadblock_right = Wedge((Para.RIGHT_X, Para.RIGHT_Y), Para.ROADBLOCK_RADIUS, 90, -90, color='green',
                                    alpha=0.7)
            ax.add_patch(roadblock_right)

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
                    ax.add_patch(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color,facecolor=item_color, angle=-(90 - a), zorder=30))
                else:
                    ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                    ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                    ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                    ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

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

            obses = self._convert_to_abso(self.obses)
            obses = obses.numpy()
            obses_ego, obses_track, obses_light, obses_task, obses_ref, obses_his_ac, obses_other = self._split_all(obses)

            # plot others
            for index in range(self.veh_num + self.bike_num + self.person_num):
                item = obses_other[:, self.per_other_info_dim * index:self.per_other_info_dim * (index + 1)]
                other_x, other_y, other_v, other_phi, other_l, other_w = item[:, 0], item[:, 1], item[:, 2], item[:, 3], item[:, 4], item[:, 5]
                if index < self.bike_num:
                    type = 'bicycle_1'
                elif self.bike_num <= index < self.person_num + self.bike_num:
                    type = 'DEFAULT_PEDTYPE'
                else:
                    type = 'veh'
                if 1:
                    plot_phi_line(type, other_x, other_y, other_phi, 'black')
                    draw_rotate_rec(type, other_x, other_y, other_phi, other_l, other_w, 'black')

            # plot own car
            delta_lat, delta_phi, delta_v = np.squeeze(obses_track, axis=0)
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = np.squeeze(obses_ego, axis=0)

            plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec("self_car", ego_x, ego_y, ego_phi, Para.L, Para.W, 'red')

            # plot text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_lat: {:.2f}m'.format(delta_lat))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            # plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

            if self.actions is not None:
                steer, a_x = self.actions[0, 0], self.actions[0, 1]
                plt.text(text_x, text_y_start - next(ge),
                         r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val[0]))

            plt.show()
            plt.pause(0.1)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class ReferencePath(object):
    def __init__(self, task, green_or_red='green'):
        self.task = task
        self.path_list = {}
        self.path_len_list = []
        self.control_points = []
        self.green_or_red = green_or_red
        self._construct_ref_path(self.task)
        self.path = None
        self.ref_encoding = None
        self.set_path(green_or_red)

    def set_path(self, green_or_red='green', path_index=None):
        if path_index is None:
            path_index = np.random.choice(len(self.path_list[self.green_or_red]))
        self.ref_encoding = REF_ENCODING[path_index]
        self.path = self.path_list[green_or_red][path_index]

    def get_future_n_point(self, ego_x, ego_y, n, dt=0.1):  # not include the current closest point
        idx, _ = self._find_closest_point(ego_x, ego_y)
        future_n_x, future_n_y, future_n_phi, future_n_v = [], [], [], []
        for _ in range(n):
            x, y, phi, v = self.idx2point(idx)
            ds = v * dt
            s = 0
            while s < ds:
                if idx + 1 >= len(self.path[0]):
                    break
                next_x, next_y, _, _ = self.idx2point(idx + 1)
                s += np.sqrt(np.square(next_x - x) + np.square(next_y - y))
                x, y = next_x, next_y
                idx += 1
            x, y, phi, v = self.idx2point(idx)
            future_n_x.append(x)
            future_n_y.append(y)
            future_n_phi.append(phi)
            future_n_v.append(v)
        future_n_point = np.stack([np.array(future_n_x, dtype=np.float32), np.array(future_n_y, dtype=np.float32),
                                   np.array(future_n_phi, dtype=np.float32), np.array(future_n_v, dtype=np.float32)],
                                  axis=0)
        return future_n_point

    def tracking_error_vector(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        # np.sin(phi0_rad) * x - np.cos(phi0_rad) * y - np.sin(phi0_rad) * x0 + np.cos(phi0_rad) * y0 = 0
        a, b, c = (x0, y0), (x0 + cos(phi0_rad), y0 + sin(phi0_rad)), (ego_x, ego_y)
        dist_a2c = np.sqrt(np.square(ego_x - x0) + np.square(ego_y - y0))
        dist_c2line = abs(sin(phi0_rad) * ego_x - cos(phi0_rad) * ego_y - sin(phi0_rad) * x0 + cos(phi0_rad) * y0)
        signed_dist_lateral = self._judge_sign_left_or_right(a, b, c) * dist_c2line
        signed_dist_longi = self._judge_sign_ahead_or_behind(a, b, c) * np.sqrt(np.abs(dist_a2c ** 2 - dist_c2line ** 2))
        return np.array([signed_dist_longi, signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def tracking_error_vector_vectorized(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        # np.sin(phi0_rad) * x - np.cos(phi0_rad) * y - np.sin(phi0_rad) * x0 + np.cos(phi0_rad) * y0 = 0

        vector_ref_phi = np.array([np.cos(phi0_rad), np.sin(phi0_rad)])
        vector_ref_phi_ccw_90 = np.array([-np.sin(phi0_rad), np.cos(phi0_rad)]) # ccw for counterclockwise
        vector_ego2ref = np.array([x0 - ego_x, y0 - ego_y])

        signed_dist_longi = np.negative(np.dot(vector_ego2ref, vector_ref_phi))
        signed_dist_lateral = np.negative(np.dot(vector_ego2ref, vector_ref_phi_ccw_90))

        return np.array([signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def idx2point(self, idx):
        return self.path[0][idx], self.path[1][idx], self.path[2][idx], self.path[3][idx]

    def _judge_sign_left_or_right(self, a, b, c):
        # see https://www.cnblogs.com/manyou/archive/2012/02/23/2365538.html for reference
        # return +1 for left and -1 for right in our case
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        featured = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
        if abs(featured) < 1e-8:
            return 0.
        else:
            return featured / abs(featured)

    def _judge_sign_ahead_or_behind(self, a, b, c):
        # return +1 if head else -1
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        vector1 = np.array([x2 - x1, y2 - y1])
        vector2 = np.array([x3 - x1, y3 - y1])
        mul = np.sum(vector1 * vector2)
        if abs(mul) < 1e-8:
            return 0.
        else:
            return mul / np.abs(mul)

    def get_bezier_control_points(self, x1, y1, phi1, x4, y4, phi4):
        # weight = 1/2
        # weight = 3/5
        weight = 7/10
        x2 = x1 * ((cos(phi1) ** 2) * (1-weight) + sin(phi1) ** 2) + y1 * (-sin(phi1) * cos(phi1) * weight) + x4 * ((cos(phi1) ** 2) * weight) + y4 * (sin(phi1) * cos(phi1) * weight)
        y2 = x1 * (-sin(phi1) * cos(phi1) * weight) + y1 * (cos(phi1) ** 2 + (sin(phi1) ** 2) * (1-weight)) + x4 * (sin(phi1) * cos(phi1) * weight) + y4 * ((sin(phi1) ** 2) * weight)
        x3 = x1 * (cos(phi4) ** 2) * weight + y1 * (sin(phi4) * cos(phi4) * weight) + x4 * ((cos(phi4) ** 2) * (1-weight) + sin(phi4) ** 2) + y4 * (-sin(phi4) * cos(phi4) * weight)
        y3 = x1 * (sin(phi4) * cos(phi4) * weight) + y1 * ((sin(phi4) ** 2) * weight) + x4 * (-sin(phi4) * cos(phi4) * weight) + y4 * (cos(phi4) ** 2 + (sin(phi4) ** 2) * (1-weight))
        control_point2 = x2, y2
        control_point3 = x3, y3
        return control_point2, control_point3

    def _construct_ref_path(self, task):
        sl = 40  # straight length
        dece_dist = 20
        meter_pointnum_ratio = 30
        planed_trj_g = []
        planed_trj_r = []
        if task == 'left':
            lane_width_flag = [Para.L_OUT_0, Para.L_OUT_1, Para.L_OUT_2]
            start_xs = [Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 * 0.5) * sin(Para.ANGLE_D * pi / 180)]
            start_ys = [Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 * 0.5) * cos(Para.ANGLE_D * pi / 180)]
            end_xs = [-Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH] * Para.LANE_NUMBER_LAT_OUT
            end_ys = [Para.OFFSET_L + Para.L_GREEN + sum(lane_width_flag[:i]) + 0.5 * lane_width_flag[i] for i in range(Para.LANE_NUMBER_LAT_OUT)]
            for start_x in start_xs:
                for end_y in end_ys:
                    control_point1 = start_x, start_ys[start_xs.index(start_x)]     # 左转第一个控制点
                    control_point4 = end_xs[end_ys.index(end_y)], end_y     # 左转第四个控制点
                    control_point2, control_point3 = self.get_bezier_control_points(control_point1[0], control_point1[1], Para.ANGLE_D*pi/180, control_point4[0], control_point4[1], pi)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                        dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = np.linspace(start_x - sl * cos(Para.ANGLE_D*pi/180), start_x, int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(start_ys[start_xs.index(start_x)] - sl * sin(Para.ANGLE_D*pi/180), start_ys[start_xs.index(start_x)], int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(end_xs[end_ys.index(end_y)], end_xs[end_ys.index(end_y)] - sl - Para.WALK_WIDTH, (sl + int(Para.WALK_WIDTH)) * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_y * np.ones(shape=((sl + int(Para.WALK_WIDTH)) * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                    vs_green = np.array([5.0] * len(start_straight_line_x) + [4.0] * (len(trj_data[0]) - 1) + [5.0] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([5.0] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L/2))), dtype=np.float32)
                    vs_red_1 = np.linspace(5.0, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.array([0.0] * (meter_pointnum_ratio * int(Para.L/2) + len(trj_data[0]) - 1) + [0.0] * len(end_straight_line_x), dtype=np.float32)
                    vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        elif task == 'straight':
            lane_width_flag = [Para.U_OUT_0, Para.U_OUT_1]
            start_xs = [Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 * 0.5) * sin(Para.ANGLE_D * pi / 180)]
            start_ys = [Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 * 0.5) * cos(Para.ANGLE_D * pi / 180)]
            end_xs = [Para.OFFSET_U_X + sum(lane_width_flag[:i]) + 0.5 * lane_width_flag[i] for i in range(Para.LANE_NUMBER_LON_OUT)]
            end_ys = [Para.OFFSET_U_Y - Para.U_OUT_0 * 0.5 * cos(Para.ANGLE_U*pi/180), Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1 * 0.5) * cos(Para.ANGLE_U * pi / 180)]
            for start_x in start_xs:
                for end_x in end_xs:
                    control_point1 = start_x, start_ys[start_xs.index(start_x)]     # 直行第一个控制点
                    control_point4 = end_x, end_ys[end_xs.index(end_x)]     # 直行第四个控制点
                    control_point2, control_point3 = self.get_bezier_control_points(control_point1[0], control_point1[1], Para.ANGLE_D * pi / 180, control_point4[0], control_point4[1], Para.ANGLE_U * pi / 180)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                        , dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = np.linspace(start_x - sl * cos(Para.ANGLE_D * pi / 180), start_x, int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(start_ys[start_xs.index(start_x)] - sl * sin(Para.ANGLE_D*pi/180), start_ys[start_xs.index(start_x)], int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(end_x, end_x + sl * cos(Para.ANGLE_U * pi / 180), int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    end_straight_line_y = np.linspace(end_ys[end_xs.index(end_x)], end_ys[end_xs.index(end_x)] + sl * sin(Para.ANGLE_U*pi/180), int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                    vs_green = np.array([5.0] * len(start_straight_line_x) + [4.0] * (len(trj_data[0]) - 1) + [5.0] *
                                        len(end_straight_line_x), dtype=np.float32)
                    vs_red_0 = np.array([5.0] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L/2))),
                        dtype=np.float32)
                    vs_red_1 = np.linspace(5.0, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                    vs_red_2 = np.array([0.0] * (meter_pointnum_ratio * int(Para.L/2) + len(trj_data[0]) - 1) + [0.0] * len(end_straight_line_x), dtype=np.float32)
                    vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_red
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        else:
            assert task == 'right'
            lane_width_flag = [Para.R_OUT_0, Para.R_OUT_1, Para.R_OUT_2]
            start_xs = [Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 * 0.5) * sin(Para.ANGLE_D * pi / 180)]
            start_ys = [Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1 * 0.5) * cos(Para.ANGLE_D * pi / 180)]
            end_xs = [Para.CROSSROAD_SIZE_LAT / 2] * Para.LANE_NUMBER_LAT_OUT
            end_ys = [Para.OFFSET_R - sum(lane_width_flag[:i]) - 0.5 * lane_width_flag[i] for i in range(Para.LANE_NUMBER_LAT_OUT)]
            for start_x in start_xs:
                for end_y in end_ys:
                    control_point1 = start_x, start_ys[start_xs.index(start_x)]     # 右转第一个控制点
                    control_point4 = end_xs[end_ys.index(end_y)], end_y     # 右转第四个控制点
                    control_point2, control_point3 = self.get_bezier_control_points(control_point1[0], control_point1[1], Para.ANGLE_D * pi / 180, control_point4[0], control_point4[1], 0)
                    self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                    node = np.asfortranarray(
                        [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                         [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                        dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = np.linspace(start_x - sl * cos(Para.ANGLE_D*pi/180), start_x, int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(start_ys[start_xs.index(start_x)] - sl * sin(Para.ANGLE_D*pi/180), start_ys[start_xs.index(start_x)], int(sl * meter_pointnum_ratio), dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(end_xs[end_ys.index(end_y)], end_xs[end_ys.index(end_y)] + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_y * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                    vs_green = np.array([5.0] * len(start_straight_line_x) + [4.0] * (len(trj_data[0]) - 1) + [5.0] *
                                        len(end_straight_line_x), dtype=np.float32)
                    planed_trj_green = xs_1, ys_1, phis_1, vs_green
                    planed_trj_red = xs_1, ys_1, phis_1, vs_green  # the same velocity design for turning right
                    planed_trj_g.append(planed_trj_green)
                    planed_trj_r.append(planed_trj_red)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

    def _find_closest_point(self, x, y, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        dists = np.square(x - reduced_path_x) + np.square(y - reduced_path_y)
        idx = np.argmin(dists) * ratio
        return idx, self.idx2point(idx)

    def plot_path(self, x, y):
        plt.axis('equal')
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, path in enumerate(self.path_list['green']):
            plt.plot(path[0], path[1], color=color[i], alpha=1.0)

        for _, point in enumerate(self.control_points):
            for item in point:
                plt.scatter(item[0], item[1], color='red')
        print(self.path_len_list)

        index, closest_point = self._find_closest_point(np.array([x], np.float32),
                                                        np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('right', 0)
    path.plot_path(1.875, 0)


def test_future_n_data():
    path = ReferencePath('straight', '0')
    plt.axis('equal')
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], 'go')
    for point in future_data_list:
        plt.plot(point[0], point[1], 'r*')
    plt.show()


def test_compute_next_track_info():
    model = EnvironmentModel()
    next_ego_infos = np.array([[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array([[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2 + 5, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array([[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2 - 5, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array([[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 8, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array([[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 15, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

def test_tracking_error_vector():
    # path = ReferencePath('straight', green_or_red='green')
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, -3.625,  0, 1.67])
    #
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 * 3, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, 4.5,  0, 1.67])
    #
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_1 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, -0.125,  0, 1.67])

    path = ReferencePath('left', green_or_red='green')
    x, y, phi, v = -Para.CROSSROAD_SIZE_LAT/2 - 10, Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 / 2, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, -7.5,  0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    x, y, phi, v = -Para.CROSSROAD_SIZE_LAT/2 - 10, Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, 9.375,  0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    path = ReferencePath('right', green_or_red='green')
    x, y, phi, v = Para.CROSSROAD_SIZE_LAT/2, Para.OFFSET_R - Para.LANE_WIDTH_1 / 2, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.0, -7.5,  0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    x, y, phi, v = Para.CROSSROAD_SIZE_LAT/2, Para.OFFSET_R - Para.LANE_WIDTH_1 *3, 0, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, 9.375,  0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)


def test_model():
    from endtoend import CrossroadEnd2endMix
    env = CrossroadEnd2endMix()
    model = EnvironmentModel()
    while 1:
        obs, info = env.reset()
        for i in range(35):
            obs_list, future_point_list = [], []
            obs_list.append(obs)
            future_point_list.append(info['future_n_point'])
            action = np.array([0, -1], dtype=np.float32)
            # obs, reward, done, info = env.step(action)
            env.render()
            obses = np.stack(obs_list, 0)
            future_points = np.array(future_point_list)
            model.reset(obses)
            print(obses.shape, future_points.shape)
            for rollout_step in range(10):
                actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
                obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
                    veh2bike4real, veh2person4real = model.rollout_out(actions, future_points[:, :, i])
                model.render()


def test_ref():
    import numpy as np
    import matplotlib.pyplot as plt
    # ref = ReferencePath('left')
    # path1, path2, path3 = ref.path_list
    # path1, path2, path3 = [ite[1200:-1200] for ite in path1],\
    #                       [ite[1200:-1200] for ite in path2], \
    #                       [ite[1200:-1200] for ite in path3]
    # x1, y1, phi1 = path1
    # x2, y2, phi2 = path2
    # x3, y3, phi3 = path3
    # p1, p2, p3 = np.arctan2(y1-(-CROSSROAD_SIZE/2), x1 - (-CROSSROAD_SIZE/2)), \
    #              np.arctan2(y2 - (-CROSSROAD_SIZE / 2), x2 - (-CROSSROAD_SIZE / 2)), \
    #              np.arctan2(y3 - (-CROSSROAD_SIZE / 2), x3 - (-CROSSROAD_SIZE / 2))
    # d1, d2, d3 = np.sqrt(np.square(x1-(-CROSSROAD_SIZE/2))+np.square(y1-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x2-(-CROSSROAD_SIZE/2))+np.square(y2-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x3-(-CROSSROAD_SIZE/2))+np.square(y3-(-CROSSROAD_SIZE/2)))
    #
    # plt.plot(p1, d1, 'r')
    # plt.plot(p2, d2, 'g')
    # plt.plot(p3, d3, 'b')
    # z1 = np.polyfit(p1, d1, 3, rcond=None, full=False, w=None, cov=False)
    # p1_fit = np.poly1d(z1)
    # plt.plot(p1, p1_fit(p1), 'r*')
    #
    # z2 = np.polyfit(p2, d2, 3, rcond=None, full=False, w=None, cov=False)
    # p2_fit = np.poly1d(z2)
    # plt.plot(p2, p2_fit(p2), 'g*')
    #
    # z3 = np.polyfit(p3, d3, 3, rcond=None, full=False, w=None, cov=False)
    # p3_fit = np.poly1d(z3)
    # plt.plot(p3, p3_fit(p3), 'b*')

    ref = ReferencePath('left', '0')
    # print(ref.path_list[ref.judge_traffic_light('0')])
    path1, path2, path3 = ref.path_list[LIGHT[0]]
    path1, path2, path3 = [ite[1200:-1200] for ite in path1], \
                          [ite[1200:-1200] for ite in path2], \
                          [ite[1200:-1200] for ite in path3]
    x1, y1, phi1, v1 = path1
    x2, y2, phi2, v1 = path2
    x3, y3, phi3, v1 = path3

    plt.plot(y1, x1, 'r')
    plt.plot(y2, x2, 'g')
    plt.plot(y3, x3, 'b')
    z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    print(type(list(z1)))
    p1_fit = np.poly1d(z1)
    print(z1, p1_fit)
    plt.plot(y1, p1_fit(y1), 'r*')
    plt.show()


if __name__ == '__main__':
    # test_model()
    test_tracking_error_vector()
    # test_compute_next_track_info()
