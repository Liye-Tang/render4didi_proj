#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend_env_utils.py
# =====================================

import math
import os
import numpy as np
from collections import OrderedDict
from shapely import geometry


class ActionStore(list):
    def __init__(self, maxlen=2):
        super(ActionStore, self).__init__()
        self.maxlen = maxlen
        for _ in range(maxlen):
            self.append(np.zeros(2, dtype=np.float32))
    
    def put(self, action):
        assert len(self) == self.maxlen
        self.pop(0)
        self.append(action)

    def reset(self):
        for i in range(self.maxlen):
            self[i] = np.zeros(2, dtype=np.float32)

    
class Para:
    # N(left to right)(1-10)：
    N_I = [[-22585.56, 6952.40], [-22583.76, 6951.73], [-22582.49, 6951.69], [-22579.32, 6950.42], [-22576.35, 6949.04],
           [-22575.04, 6948.30], [-22572.23, 6947.10], [-22568.94, 6945.69], [-22567.95, 6945.23], [-22566.09, 6944.48]]
    N_O = [[-22577.3, 6972.54], [-22575.4, 6971.89], [-22574.5, 6971.56], [-22571.1, 6970.33], [-22568.0, 6969.13],
           [-22566.7, 6968.75], [-22563.5, 6967.5], [-22560.3, 6966.32], [-22559.4, 6965.97], [-22557.6, 6965.27]]
    # S(left to right)(1-5)：
    S_I = [[-22605.54, 6893.55], [-22602.10, 6892.21], [-22598.56, 6890.39], [-22594.67, 6888.74], [-22591.33, 6887.24]]
    S_O = [[-22612.16, 6878.01], [-22608.71, 6876.58], [-22605.09, 6875.08], [-22601.20, 6873.40], [-22597.78, 6871.96]]
    # W(up to down)(1-9)：
    W_I = [[-22604.56, 6944.07], [-22605.89, 6940.73], [-22606.83, 6937.10], [-22607.81, 6934.05], [-22610.35, 6924.49],
           [-22611.39, 6921.26], [-22612.55, 6917.94], [-22613.82, 6914.50], [-22614.78, 6910.95]]
    W_O = [[-22625.47, 6951.37], [-22626.46, 6947.91], [-22627.42, 6944.30], [-22628.55, 6941.23], [-22640.24, 6934.84],
           [-22640.65, 6931.46], [-22644.06, 6928.82], [-22643.34, 6924.73], [-22644.22, 6921.27]]
    # E(up to down)(1-9)：
    E_I = [[-22558.54, 6928.29], [-22559.33, 6924.80], [-22560.49, 6921.22], [-22561.70, 6917.87], [-22562.78, 6914.85],
           [-22567.85, 6906.15], [-22568.97, 6903.05], [-22570.08, 6899.55], [-22571.34, 6895.99]]
    E_O = [[-22534.59, 6919.89], [-22535.88, 6916.50], [-22536.87, 6912.87], [-22538.00, 6909.60], [-22538.79, 6906.43],
           [-22544.10, 6897.77], [-22545.03, 6894.59], [-22546.14, 6891.03], [-22547.37, 6887.58]]

    def calculate_distance_angle(point_A, point_B):
        # Return the distance and angle between B-A and the positive x-axis.
        ax, ay = point_A[0], point_A[1]
        bx, by = point_B[0], point_B[1]
        distance = math.hypot(bx - ax, by - ay)
        angle = math.atan2(by - ay, bx - ax) * 180 / math.pi
        return distance, angle

    def get_point_line_distance(point, line):
        point_x = point[0]
        point_y = point[1]
        line_s_x = line[0][0]
        line_s_y = line[0][1]
        line_e_x = line[1][0]
        line_e_y = line[1][1]
        if line_e_x - line_s_x == 0:
            return math.fabs(point_x - line_s_x)
        if line_e_y - line_s_y == 0:
            return math.fabs(point_y - line_s_y)
        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        b = line_s_y - k * line_s_x
        dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
        return dis

    # MAP
    L, W = 4.8, 2.0
    WALK_WIDTH = 6.00
    BIKE_LANE_WIDTH = 1.0
    PERSON_LANE_WIDTH = 2.0

    OFFSET_L = -1.7
    OFFSET_R = -5.2
    OFFSET_U_X = 1.65
    OFFSET_U_Y = 33.2
    OFFSET_D_X = -1.07
    OFFSET_D_Y = -29.60

    # N----D
    D_IN_0, _ = calculate_distance_angle(N_I[3], N_I[4])
    D_IN_1, _ = calculate_distance_angle(N_I[2], N_I[3])
    D_GREEN, _ = calculate_distance_angle(N_I[4], N_I[5])
    D_OUT_0, _ = calculate_distance_angle(N_I[5], N_I[6])
    D_OUT_1, _ = calculate_distance_angle(N_I[6], N_I[7])
    _, D_ANGLE1 = calculate_distance_angle(N_I[5], N_O[5])
    _, D_ANGLE2 = calculate_distance_angle(E_I[4], E_O[4])
    ANGLE_D = D_ANGLE1 - D_ANGLE2

    # S----U
    U_IN_0, _ = calculate_distance_angle(S_I[2], S_I[3])
    U_IN_1, _ = calculate_distance_angle(S_I[3], S_I[4])
    U_OUT_0, _ = calculate_distance_angle(S_I[1], S_I[2])
    U_OUT_1, _ = calculate_distance_angle(S_I[0], S_I[1])
    _, U_ANGLE1 = calculate_distance_angle(S_O[2], S_I[2])
    _, U_ANGLE2 = calculate_distance_angle(W_O[4], W_I[4])
    ANGLE_U = U_ANGLE1 - U_ANGLE2

    # W----R
    R_IN_0, _ = calculate_distance_angle(W_I[4], W_I[5])
    R_IN_1, _ = calculate_distance_angle(W_I[5], W_I[6])
    R_IN_2, _ = calculate_distance_angle(W_I[6], W_I[7])
    R_IN_3, _ = calculate_distance_angle(W_I[7], W_I[8])
    R_GREEN, _ = calculate_distance_angle(W_I[3], W_I[4])
    R_GREEN -= 0.1
    R_OUT_0, _ = calculate_distance_angle(W_I[2], W_I[3])
    R_OUT_1, _ = calculate_distance_angle(W_I[1], W_I[2])
    R_OUT_2, _ = calculate_distance_angle(W_I[0], W_I[1])

    # E----L
    L_IN_0, _ = calculate_distance_angle(E_I[3], E_I[4])
    L_IN_1, _ = calculate_distance_angle(E_I[2], E_I[3])
    L_IN_2, _ = calculate_distance_angle(E_I[1], E_I[2])
    L_IN_3, _ = calculate_distance_angle(E_I[0], E_I[1])
    L_IN_3 -= 0.1
    L_GREEN, _ = calculate_distance_angle(E_I[4], E_I[5])
    L_GREEN -= 0.2
    L_OUT_0, _ = calculate_distance_angle(E_I[5], E_I[6])
    L_OUT_1, _ = calculate_distance_angle(E_I[6], E_I[7])
    L_OUT_2, _ = calculate_distance_angle(E_I[7], E_I[8])

    CROSSROAD_SIZE_LAT = get_point_line_distance(W_I[5], [E_I[1], E_I[4]]) + WALK_WIDTH * 2

    LANE_NUMBER_LON_IN = 2
    LANE_NUMBER_LON_OUT = 2
    LANE_NUMBER_LAT_IN = 4
    LANE_NUMBER_LAT_OUT = 3

    #roadblock
    ROADBLOCK_RADIUS = 4.5
    LEFT_X = -CROSSROAD_SIZE_LAT / 2 + WALK_WIDTH
    LEFT_Y = OFFSET_L + L_GREEN / 2
    RIGHT_X = CROSSROAD_SIZE_LAT / 2 - WALK_WIDTH
    RIGHT_Y = OFFSET_R + R_GREEN / 2

    # DIM
    EGO_ENCODING_DIM = 6
    TRACK_ENCODING_DIM = 3
    LIGHT_ENCODING_DIM = 2
    TASK_ENCODING_DIM = 3
    REF_ENCODING_DIM = 3
    HIS_ACT_ENCODING_DIM = 4
    PER_OTHER_INFO_DIM = 10

    # MAX NUM
    MAX_VEH_NUM = 6  # to be align with VEHICLE_MODE_DICT
    MAX_BIKE_NUM = 6  # to be align with BIKE_MODE_DICT
    MAX_PERSON_NUM = 6  # to be align with PERSON_MODE_DICT

    # NOISE
    # (v_x, v_y, r, x, y, phi) for ego
    # (x, y, v, phi, l, w; type encoding (d=3), turn rad) for other
    EGO_MEAN = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
    # EGO_VAR = np.diag([0.0418, 0.0418, 0., 0.0245, 0.0227, 0.0029*(180./np.pi)**2]).astype(np.float32)
    EGO_VAR = np.diag([0., 0., 0., 0., 0., 0.]).astype(np.float32)

    VEH_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_VEH_NUM)
    VEH_VAR = np.tile(np.array([0.0245, 0.0227, 0.0418, 0.0029*(180./np.pi)**2, 0.0902, 0.0202, 0., 0., 0., 0.,], dtype=np.float32), MAX_VEH_NUM)

    BIKE_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_BIKE_NUM)
    BIKE_VAR = np.tile(np.array([0.172**2, 0.1583**2, 0.1763**2, (0.1707*180./np.pi)**2, 0.1649**2, 0.1091**2, 0., 0., 0., 0.,], dtype=np.float32), MAX_BIKE_NUM)

    PERSON_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_PERSON_NUM)
    PERSON_VAR = np.tile(np.array([0.1102**2, 0.1108**2, 0.1189**2, (0.2289*180./np.pi)**2, 0.1468**2, 0.1405**2, 0., 0., 0., 0.,], dtype=np.float32), MAX_PERSON_NUM)

    OTHERS_MEAN = np.concatenate([BIKE_MEAN, PERSON_MEAN, VEH_MEAN], axis=-1) # order determined in line 735 in e2e.py
    OTHERS_VAR = np.diag(np.concatenate([BIKE_VAR, PERSON_VAR, VEH_VAR], axis=-1)).astype(np.float32)



LIGHT_PHASE_TO_GREEN_OR_RED = {0: 'green', 1: 'green', 2: 'red', 3: 'red',
                                 4: 'red', 5: 'red', 6: 'red', 7: 'red', 8: 'red', 9: 'red'}  # 0: green, 1: red
TASK_ENCODING = dict(left=[1.0, 0.0, 0.0], straight=[0.0, 1.0, 0.0], right=[0.0, 0.0, 1.0])
LIGHT_ENCODING = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0], 4: [0.0, 1.0], 5: [0.0, 1.0],
                  6: [0.0, 1.0], 7: [0.0, 1.0], 8: [0.0, 1.0], 9: [0.0, 1.0]}
REF_ENCODING = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0]}

SUMOCFG_DIR = os.path.dirname(__file__) + "/sumo_files/cross.sumocfg"
VEHICLE_MODE_DICT = dict(left=OrderedDict(dl=2, du=2, ud=2, ul=2),
                         straight=OrderedDict(dl=2, du=2, ru=2, ur=2),
                         right=OrderedDict(dr=2, du=2, ur=2, lr=2))
BIKE_MODE_DICT = dict(left=OrderedDict(ud_b=4),
                      straight=OrderedDict(du_b=4),
                      right=OrderedDict(du_b=2, lr_b=2))  # 2 0
PERSON_MODE_DICT = dict(left=OrderedDict(c3=4),
                        straight=OrderedDict(c2=4),  # 0
                        right=OrderedDict(c1=4, c2=0))

MODE2STEP = {'green_veh_left_1': 2+8+np.random.random() * 20, 'green_veh_left_2': 0+8+np.random.random() * 20,
             'green_veh_left_3': 7+8+np.random.random() * 20, 'green_veh_left_4': 7+6+np.random.random() * 20,
             'green_veh_right_1': 3+9+np.random.random() * 20, 'green_veh_right_2': 12+5+np.random.random() * 20,
             'green_veh_straight_1': 5+9+np.random.random() * 20, 'green_veh_straight_2': 9+np.random.random() * 20,
             'green_veh_straight_3': 4+6+np.random.random() * 20, 'green_veh_straight_4': 3+6+np.random.random() * 20,
             'green_ego_left_1': 0, 'green_ego_right_1': 0, 'green_ego_straight_1': 0}

MODE2STEP_TEST = {'green_ego_left_1': 0+5, 'green_veh_left_1': 2+8, 'green_veh_left_2': 0+8,
                  'green_veh_left_3': 7+8, 'green_veh_left_4': 7+6, 'green_ego_right_1': 5,
                  'green_ego_straight_1': 5, 'green_veh_right_1': 3+9, 'green_veh_right_2': 12+5,
                  'green_veh_straight_1': 5+9, 'green_veh_straight_2': 9, 'green_veh_straight_3': 4+6,
                  'green_veh_straight_4': 3+6}

MODE2INDEX_TEST = {'green_ego_left_1': 500, 'green_veh_left_1': 500, 'green_veh_left_2': 450,
                   'green_veh_left_3': 600, 'green_veh_left_4': 550, 'green_ego_right_1': 500,
                   'green_ego_straight_1': 450, 'green_veh_right_1': 500, 'green_veh_right_2': 500,
                   'green_veh_straight_1': 550, 'green_veh_straight_2': 550, 'green_veh_straight_3': 500,
                   'green_veh_straight_4': 500}

def dict2flat(inp):
    out = []
    for key, val in inp.items():
        out.extend([key] * val)
    return out


def dict2num(inp):
    out = 0
    for _, val in inp.items():
        out += val
    return out


VEHICLE_MODE_LIST = dict(left=dict2flat(VEHICLE_MODE_DICT['left']),
                         straight=dict2flat(VEHICLE_MODE_DICT['straight']),
                         right=dict2flat(VEHICLE_MODE_DICT['right']))
BIKE_MODE_LIST = dict(left=dict2flat(BIKE_MODE_DICT['left']),
                      straight=dict2flat(BIKE_MODE_DICT['straight']),
                      right=dict2flat(BIKE_MODE_DICT['right']))
PERSON_MODE_LIST = dict(left=dict2flat(PERSON_MODE_DICT['left']),
                        straight=dict2flat(PERSON_MODE_DICT['straight']),
                        right=dict2flat(PERSON_MODE_DICT['right']))

ROUTE2MODE = {('1o', '2i'): 'dr', ('1o', '3i'): 'du', ('1o', '4i'): 'dl',
              ('2o', '1i'): 'rd', ('2o', '3i'): 'ru', ('2o', '4i'): 'rl',
              ('3o', '1i'): 'ud', ('3o', '2i'): 'ur', ('3o', '4i'): 'ul',
              ('4o', '1i'): 'ld', ('4o', '2i'): 'lr', ('4o', '3i'): 'lu'}

MODE2TASK = {'dr': 'right', 'du': 'straight', 'dl': 'left',
             'rd': 'left', 'ru': 'right', 'rl': ' straight',
             'ud': 'straight', 'ur': 'left', 'ul': 'right',
             'ld': 'right', 'lr': 'straight', 'lu': 'left',
             'ud_b': 'straight', 'du_b': 'straight', 'lr_b': 'straight',
             'c1': 'straight', 'c2': 'straight', 'c3': 'straight'}

TASK2ROUTEID = {'left': 'dl', 'straight': 'du', 'right': 'dr'}

MODE2ROUTE = {'dr': ('1o', '2i'), 'du': ('1o', '3i'), 'dl': ('1o', '4i'),
              'rd': ('2o', '1i'), 'ru': ('2o', '3i'), 'rl': ('2o', '4i'),
              'ud': ('3o', '1i'), 'ur': ('3o', '2i'), 'ul': ('3o', '4i'),
              'ld': ('4o', '1i'), 'lr': ('4o', '2i'), 'lu': ('4o', '3i')}


def judge_feasible(orig_x, orig_y, task):  # map dependant
    def is_in_straight_before1(orig_x, orig_y):
        orig_x_trans, orig_y_trans, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
        OFFSET_D_X_trans, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)
        return OFFSET_D_X_trans + Para.D_GREEN < orig_x_trans < OFFSET_D_X_trans + Para.D_GREEN + Para.D_IN_0 and orig_y_trans <= OFFSET_D_Y_trans

    def is_in_straight_before2(orig_x, orig_y):
        orig_x_trans, orig_y_trans, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
        OFFSET_D_X_trans, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)
        return OFFSET_D_X_trans + Para.D_GREEN + Para.D_IN_0 < orig_x_trans < OFFSET_D_X_trans + Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1\
               and orig_y_trans <= OFFSET_D_Y_trans

    def is_in_straight_after(orig_x, orig_y):
        orig_x_trans, orig_y_trans, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_U-90)
        OFFSET_U_X_trans, OFFSET_U_Y_trans, _ = rotate_coordination(Para.OFFSET_U_X, Para.OFFSET_U_Y, 0, Para.ANGLE_U - 90)
        return OFFSET_U_X_trans < orig_x_trans < OFFSET_U_X_trans + Para.U_OUT_0 + Para.U_OUT_1 and orig_y_trans >= OFFSET_U_Y_trans

    def is_in_left(orig_x, orig_y):
        return Para.OFFSET_L + Para.L_GREEN < orig_y < Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 and orig_x < -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH

    def is_in_right(orig_x, orig_y):
        return Para.OFFSET_R - Para.R_OUT_0 - Para.R_OUT_1 - Para.R_OUT_2 < orig_y < Para.OFFSET_R and orig_x > Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH

    # def is_in_middle(orig_x, orig_y):
    #     _, orig_y_trans_D, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
    #     _, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)
    #     _, orig_y_trans_U, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_U-90)
    #     _, OFFSET_U_Y_trans, _ = rotate_coordination(Para.OFFSET_U_X, Para.OFFSET_U_Y, 0, Para.ANGLE_U - 90)
    #     return True if OFFSET_D_Y_trans < orig_y_trans_D  and orig_y_trans_U < OFFSET_U_Y_trans and -Para.CROSSROAD_SIZE_LAT / 2 < orig_x < Para.CROSSROAD_SIZE_LAT / 2 else False

    def is_in_middle_left(orig_x, orig_y):
        _, orig_y_trans_D, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
        _, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)

        dis_left = np.sqrt(np.square(orig_x - Para.LEFT_X) + np.square(orig_y - Para.LEFT_Y))
        return True if orig_y_trans_D > OFFSET_D_Y_trans \
                       and orig_y < Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1 + Para.L_OUT_2 \
                       + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH \
                       and -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH < orig_x < Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH and \
                       dis_left > Para.ROADBLOCK_RADIUS else False

    def is_in_middle_straight(orig_x, orig_y):
        orig_x_trans_D, orig_y_trans_D, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
        OFFSET_D_X_trans, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)
        orig_x_trans_U, orig_y_trans_U, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_U-90)
        OFFSET_U_X_trans, OFFSET_U_Y_trans, _ = rotate_coordination(Para.OFFSET_U_X, Para.OFFSET_U_Y, 0, Para.ANGLE_U - 90)
        return True if orig_y_trans_D > OFFSET_D_Y_trans \
                       and orig_y_trans_U < OFFSET_U_Y_trans \
                       and -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH < orig_x < Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH \
                       and orig_x_trans_D > OFFSET_D_X_trans - Para.D_OUT_0 \
                       and (orig_y < Para.OFFSET_R + Para.R_GREEN + Para.R_IN_0 + Para.R_IN_1 + Para.R_IN_2 + Para.R_IN_3 + + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH + 5.0
                            or orig_x_trans_U < OFFSET_U_X_trans + Para.U_OUT_0 + Para.U_OUT_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH) else False

    def is_in_middle_right(orig_x, orig_y):
        orig_x_trans_D, orig_y_trans_D, _ = rotate_coordination(orig_x, orig_y, 0, Para.ANGLE_D-90)
        OFFSET_D_X_trans, OFFSET_D_Y_trans, _ = rotate_coordination(Para.OFFSET_D_X, Para.OFFSET_D_Y, 0, Para.ANGLE_D - 90)

        dis_right = np.sqrt(np.square(orig_x - Para.RIGHT_X) + np.square(orig_y - Para.RIGHT_Y))

        return True if orig_y_trans_D > OFFSET_D_Y_trans \
                       and orig_y < Para.OFFSET_R + Para.R_GREEN / 2 \
                       and orig_x < Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH and orig_x_trans_D > OFFSET_D_X_trans \
                       and dis_right > Para.ROADBLOCK_RADIUS else False

    if task == 'left':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle_left(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_straight_after(
            orig_x, orig_y) or is_in_middle_straight(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before2(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle_right(orig_x, orig_y) else False


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    """
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d

def rotate_coordination_vec(orig_x, orig_y, orig_d, coordi_rotate_d):
    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    transformed_d = np.where(transformed_d>180, transformed_d - 360, transformed_d)
    transformed_d = np.where(transformed_d<=-180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route, })
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics[
        'Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics


def coordination_didi2sumo(orig_x, orig_y):
    _, line_angle = Para.calculate_distance_angle(Para.E_I[4], Para.E_I[1])
    line_angle_1 = line_angle * math.pi / 180
    shift_x = Para.E_I[4][0]-2*math.cos(line_angle_1)-(Para.CROSSROAD_SIZE_LAT/2)*math.sin(line_angle_1)
    shift_y = Para.E_I[4][1]-2*math.sin(line_angle_1)+(Para.CROSSROAD_SIZE_LAT/2)*math.cos(line_angle_1)
    _, rotate_angle = Para.calculate_distance_angle(Para.E_O[4], Para.E_I[4])
    angle = rotate_angle  # check
    shifted_x, shifted_y, _ = shift_and_rotate_coordination(orig_x, orig_y, angle, shift_x, shift_y, rotate_angle)
    return shifted_x, shifted_y

def coordination_sumo2didi(orig_x, orig_y):
    _, line_angle = Para.calculate_distance_angle(Para.E_I[4], Para.E_I[1])
    line_angle_1 = line_angle * math.pi / 180
    shift_x = Para.E_I[4][0]-2*math.cos(line_angle_1)-(Para.CROSSROAD_SIZE_LAT/2)*math.sin(line_angle_1)
    shift_y = Para.E_I[4][1]-2*math.sin(line_angle_1)+(Para.CROSSROAD_SIZE_LAT/2)*math.cos(line_angle_1)
    _, rotate_angle = Para.calculate_distance_angle(Para.E_O[4], Para.E_I[4])
    angle = rotate_angle  # check
    shifted_x, shifted_y, _ = rotate_and_shift_coordination(orig_x, orig_y, angle, -shift_x, -shift_y, -rotate_angle)
    return shifted_x, shifted_y


def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

def isInterArea(testPoint,AreaPoint):   #testPoint为待测点[x,y]
    LBPoint = AreaPoint[0]  #AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[1]
    RTPoint = AreaPoint[2]
    RBPoint = AreaPoint[3]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0):
        return True
    else:
        return False

def xy2_edgeID_lane(x, y):
    if y < Para.OFFSET_D_Y - Para.D_GREEN * math.cos(Para.ANGLE_D*math.pi/180):
        edgeID = '1o'
        x1 = Para.OFFSET_D_X + Para.D_GREEN * math.sin(Para.ANGLE_D*math.pi/180)
        y1 = Para.OFFSET_D_Y - Para.D_GREEN * math.cos(Para.ANGLE_D*math.pi/180)
        x2 = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0) * math.sin(Para.ANGLE_D*math.pi/180)
        y2 = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0) * math.cos(Para.ANGLE_D*math.pi/180)
        x3 = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0) * math.sin(Para.ANGLE_D*math.pi/180) - 60 * math.cos(Para.ANGLE_D*math.pi/180)
        y3 = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0) * math.cos(Para.ANGLE_D*math.pi/180) - 60 * math.sin(Para.ANGLE_D*math.pi/180)
        x4 = Para.OFFSET_D_X + Para.D_GREEN * math.sin(Para.ANGLE_D*math.pi/180) - 60 * math.cos(Para.ANGLE_D*math.pi/180)
        y4 = Para.OFFSET_D_Y - Para.D_GREEN * math.cos(Para.ANGLE_D*math.pi/180) - 60 * math.sin(Para.ANGLE_D*math.pi/180)
        x5 = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.sin(Para.ANGLE_D*math.pi/180)
        y5 = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.cos(Para.ANGLE_D*math.pi/180)
        x6 = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.sin(Para.ANGLE_D*math.pi/180) - 60 * math.cos(Para.ANGLE_D*math.pi/180)
        y6 = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.cos(Para.ANGLE_D*math.pi/180) - 60 * math.sin(Para.ANGLE_D*math.pi/180)
        if isInterArea([x, y], [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]):
            lane = 3
        elif isInterArea([x, y], [[x2, y2], [x5, y5], [x6, y6], [x3, y3]]):
            lane = 2
        else:
            edgeID = '0'
            lane = 0
    elif x < -Para.CROSSROAD_SIZE_LAT / 2 and Para.OFFSET_D_Y < y < Para.OFFSET_U_Y:
        edgeID = '4i'
        if Para.OFFSET_L + Para.L_GREEN <= y < Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0:
            lane = 4
        elif Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 <= y < Para.OFFSET_L + Para.L_GREEN + Para.L_OUT_0 + Para.L_OUT_1:
            lane = 3
        else:
            lane = 2
    elif y > Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1) * math.cos(Para.ANGLE_U*math.pi/180):
        edgeID = '3i'
        x1 = Para.OFFSET_U_X
        y1 = Para.OFFSET_U_Y
        x2 = Para.OFFSET_U_X + Para.U_OUT_0 * math.sin(Para.ANGLE_U*math.pi/180)
        y2 = Para.OFFSET_U_Y - Para.U_OUT_0 * math.cos(Para.ANGLE_U*math.pi/180)
        x3 = Para.OFFSET_U_X + Para.U_OUT_0 * math.sin(Para.ANGLE_U*math.pi/180) + 60 * math.cos(Para.ANGLE_U*math.pi/180)
        y3 = Para.OFFSET_U_Y - Para.U_OUT_0 * math.cos(Para.ANGLE_U*math.pi/180) + 60 * math.sin(Para.ANGLE_U*math.pi/180)
        x4 = Para.OFFSET_U_X + 60 * math.cos(Para.ANGLE_U*math.pi/180)
        y4 = Para.OFFSET_U_Y + 60 * math.sin(Para.ANGLE_U*math.pi/180)
        x5 = Para.OFFSET_U_X + (Para.U_OUT_0 + Para.U_OUT_1) * math.sin(Para.ANGLE_U*math.pi/180)
        y5 = Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1) * math.cos(Para.ANGLE_U*math.pi/180)
        x6 = Para.OFFSET_U_X + (Para.U_OUT_0 + Para.U_OUT_1) * math.sin(Para.ANGLE_U*math.pi/180) + 60 * math.cos(Para.ANGLE_U*math.pi/180)
        y6 = Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1) * math.cos(Para.ANGLE_U*math.pi/180) + 60 * math.sin(Para.ANGLE_U*math.pi/180)
        if isInterArea([x, y], [[x1, y1], [x4, y4], [x3, y3], [x2, y2]]):
            lane = 3
        elif isInterArea([x, y], [[x2, y2], [x3, y3], [x6, y6], [x5, y5]]):
            lane = 2
        else:
            edgeID = '0'
            lane = 0
    elif x > Para.CROSSROAD_SIZE_LAT / 2 and Para.OFFSET_D_Y < y < Para.OFFSET_U_Y:
        edgeID = '2i'
        if y >= Para.OFFSET_R - Para.R_OUT_0:
            lane = 4
        elif Para.OFFSET_R - Para.R_OUT_0 - Para.R_OUT_1 <= y < Para.OFFSET_R - Para.R_OUT_0:
            lane = 3
        else:
            lane = 2
    else:
        edgeID = '0'
        lane = 0
    return edgeID, lane


class Road:
    D_X1_U = Para.OFFSET_D_X + Para.D_GREEN * math.sin(Para.ANGLE_D * math.pi / 180)
    D_Y1_U = Para.OFFSET_D_Y - Para.D_GREEN * math.cos(Para.ANGLE_D * math.pi / 180)
    D_X1_D = Para.OFFSET_D_X + Para.D_GREEN * math.sin(Para.ANGLE_D * math.pi / 180) - 60 * math.cos(Para.ANGLE_D * math.pi / 180)
    D_Y1_D = Para.OFFSET_D_Y - Para.D_GREEN * math.cos(Para.ANGLE_D * math.pi / 180) - 60 * math.sin(Para.ANGLE_D * math.pi / 180)
    D_X2_U = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0) * math.sin(Para.ANGLE_D * math.pi / 180)
    D_Y2_U = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0) * math.cos(Para.ANGLE_D * math.pi / 180)
    D_X2_D = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0) * math.sin(Para.ANGLE_D * math.pi / 180) - 60 * math.cos(Para.ANGLE_D * math.pi / 180)
    D_Y2_D = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0) * math.cos(Para.ANGLE_D * math.pi / 180) - 60 * math.sin(Para.ANGLE_D * math.pi / 180)
    D_X3_U = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.sin(Para.ANGLE_D * math.pi / 180)
    D_Y3_U = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.cos(Para.ANGLE_D * math.pi / 180)
    D_X3_D = Para.OFFSET_D_X + (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.sin(Para.ANGLE_D * math.pi / 180) - 60 * math.cos(Para.ANGLE_D * math.pi / 180)
    D_Y3_D = Para.OFFSET_D_Y - (Para.D_GREEN + Para.D_IN_0 + Para.D_IN_1) * math.cos(Para.ANGLE_D * math.pi / 180) - 60 * math.sin(Para.ANGLE_D * math.pi / 180)

    U_X1_D = Para.OFFSET_U_X
    U_Y1_D = Para.OFFSET_U_Y
    U_X1_U = Para.OFFSET_U_X + 60 * math.cos(Para.ANGLE_U*math.pi/180)
    U_Y1_U = Para.OFFSET_U_Y + 60 * math.sin(Para.ANGLE_U*math.pi/180)
    U_X2_D = Para.OFFSET_U_X + (Para.U_OUT_0 + Para.U_OUT_1) * math.sin(Para.ANGLE_U*math.pi/180)
    U_Y2_D = Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1) * math.cos(Para.ANGLE_U*math.pi/180)
    U_X2_U = Para.OFFSET_U_X + (Para.U_OUT_0 + Para.U_OUT_1) * math.sin(Para.ANGLE_U*math.pi/180) + 60 * math.cos(Para.ANGLE_U*math.pi/180)
    U_Y2_U = Para.OFFSET_U_Y - (Para.U_OUT_0 + Para.U_OUT_1) * math.cos(Para.ANGLE_U*math.pi/180) + 60 * math.sin(Para.ANGLE_U*math.pi/180)

    # line equation
    D_K1 = (D_Y1_D - D_Y1_U) / (D_X1_D - D_X1_U)
    D_B1 = (D_Y1_U - D_X1_U * (D_Y1_D - D_Y1_U) / (D_X1_D - D_X1_U))
    D_K2 = (D_Y2_D - D_Y2_U) / (D_X2_D - D_X2_U)
    D_B2 = (D_Y2_U - D_X2_U * (D_Y2_D - D_Y2_U) / (D_X2_D - D_X2_U))
    D_K3 = (D_Y3_D - D_Y3_U) / (D_X3_D - D_X3_U)
    D_B3 = (D_Y3_U - D_X3_U * (D_Y3_D - D_Y3_U) / (D_X3_D - D_X3_U))

    U_K1 = (U_Y1_D - U_Y1_U) / (U_X1_D - U_X1_U)
    U_B1 = (U_Y1_U - U_X1_U * (U_Y1_D - U_Y1_U) / (U_X1_D - U_X1_U))
    U_K2 = (U_Y2_D - U_Y2_U) / (U_X2_D - U_X2_U)
    U_B2 = (U_Y2_U - U_X2_U * (U_Y2_D - U_Y2_U) / (U_X2_D - U_X2_U))

def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi, task=None):
    if task == 'left':
        while phi <= -180:
            phi += 360
        while phi > 360:
            phi -= 360
    else:
        while phi > 180:
            phi -= 360
        while phi <= -180:
            phi += 360
    return phi


def test_action_store():
    action_store = ActionStore()
    action_store.put(np.array([1., 0.], dtype=np.float32))

    print(len(action_store))
    print(action_store)
    print(action_store[0])
    print(type(action_store[1]))

    action_store.reset()
    print(action_store)
    print(action_store[0])
    print(action_store[1])
    action_store.put(np.array([1., 0.], dtype=np.float32))
    action_store.put(np.array([1., 0.], dtype=np.float32))
    print(action_store)



if __name__ == '__main__':
    # test_action_store()
    # print(xy2_edgeID_lane(4.62,37.99))
    # point_1 = [-22612.55, 6917.94]
    # line_point_1 = [-22559.33, 6924.80]
    # line_point_2 = [-22562.78, 6914.85]
    # print(get_point_line_distance(point_1, [line_point_1, line_point_2]))
    # A = [-22614.78, 6910.95]
    # B = [-22613.82, 6914.50]
    # distance, angle = calculate_distance_angle(A, B)
    # print('distance:', distance, 'angle:', angle)
    # a = Para()
    # print(a.CROSSROAD_SIZE_LAT)
    # print('U', Para.U_IN_0, Para.U_IN_1, '------', Para.U_OUT_0, Para.U_OUT_1)
    # print('D', Para.D_IN_0, Para.D_IN_1, '------', Para.D_OUT_0, Para.D_OUT_1)
    # print('L', Para.L_IN_0, Para.L_IN_1, Para.L_IN_2, Para.L_IN_3, '------', Para.L_OUT_0, Para.L_OUT_1, Para.L_OUT_2)
    # print('R', Para.R_IN_0, Para.R_IN_1, Para.R_IN_2, Para.R_IN_3, '------', Para.R_OUT_0, Para.R_OUT_1, Para.R_OUT_2)
    # print(Para.L_GREEN, Para.R_GREEN, Para.D_GREEN)
    # print(Para.CROSSROAD_SIZE_LAT)
    # print(Para.ANGLE_U, Para.ANGLE_D)
    # print('didi2sumo', coordination_didi2sumo(Para.E_I[4][0], Para.E_I[4][1]))
    # print('sumo2didi', coordination_sumo2didi(-24.005982240251946, -2.090883080432098))
    # print(Para.E_I[4][0], Para.E_I[4][1])
    corner = ((0.11815315750125033, -48.83591669365837), (2.115684642944006, -48.93525432331027),
     (-0.12025715366330525, -53.62999225872098), (1.8772743317794505, -53.72932988837288))
    for i in corner:
        print(judge_feasible(i[0], i[1], 'straight'))
