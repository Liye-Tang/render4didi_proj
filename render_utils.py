# policy_type
IS_MIX = True

# action
if IS_MIX:
    ACC_SCALE = 1.5
    ACC_SHIFT = 0.5
    STEER_SCALE = 0.3
    STEER_SHIFT = 0
else:
    ACC_SCALE = 2.25
    ACC_SHIFT = 0.75
    STEER_SCALE = 0.4
    STEER_SHIFT = 0

# controller
STEER_RATIO = 16.6