# Copyright (C) 2018 Shadow Robot Company Ltd - All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of the content in this file, via any medium is strictly prohibited.

from smart_grasping_sandbox.smart_grasper import SmartGrasper
from tf.transformations import quaternion_from_euler
from math import pi
import time
import rospy
from math import sqrt, pow
import random
from sys import argv
import uuid

sgs = SmartGrasper()

MIN_LIFT_STEPS = 5
MAX_BALL_DISTANCE = 0.6

CLOSED_HAND = {}

CLOSED_HAND["H1_F1J1"] = 0.0
CLOSED_HAND["H1_F1J2"] = 0.25
CLOSED_HAND["H1_F1J3"] = 0.4
CLOSED_HAND["H1_F2J1"] = 0.0
CLOSED_HAND["H1_F2J2"] = 0.25
CLOSED_HAND["H1_F2J3"] = 0.4
CLOSED_HAND["H1_F3J1"] = 0.0
CLOSED_HAND["H1_F3J2"] = 0.25
CLOSED_HAND["H1_F3J3"] = 0.4

JOINT_NAMES = CLOSED_HAND.keys()


rospy.loginfo("openning")
sgs.open_hand()
time.sleep(1)
rospy.loginfo("closing")
sgs.close_hand()

import pdb; pdb.set_trace()
