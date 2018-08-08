#!/usr/bin/python

from bayes_opt import BayesianOptimization
from smart_grasping_sandbox.smart_grasper import SmartGrasper
from tf.transformations import quaternion_from_euler
from math import pi
import time
import rospy
from math import sqrt, pow
import random
from sys import argv
from numpy import var, mean
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.optimizers import sgd
import keras
import pickle



sgs = SmartGrasper()

MIN_LIFT_STEPS = 1
# Cut off the action
MAX_BALL_DISTANCE = 0.4

REPEAT_GRASP = 1

# SGS client
# Grasp using parameters fed and return stated when asked
class GraspQuality(object):

    def __init__(self, sgs):
        self.sgs = sgs
        self.last_distance = None
        self.current_grasp = {}

    def check_stable(self, joint_names):
        current_min = 500
        positions = []
        velocities = []
        efforts = []
        for k in range(30):
            sgs.move_tip(y=0.03)
            ball_distance = self.__compute_euclidean_distance()
            if k > MIN_LIFT_STEPS and ball_distance < current_min:
                current_min = ball_distance
                break
            if ball_distance > MAX_BALL_DISTANCE:
                break

            time.sleep(0.5)
        robustness = (1/(current_min - 0.18))**2
        return robustness

    def __compute_euclidean_distance(self):
        ball_pose = self.sgs.get_object_pose()
        hand_pose = self.sgs.get_tip_pose()
        dist = sqrt((hand_pose.position.x - ball_pose.position.x)**2 + \
                     (hand_pose.position.y - ball_pose.position.y)**2 + \
                     (hand_pose.position.z - ball_pose.position.z)**2)
        return dist

    def run_experiments(self, grasp_distance,
                        H1_F1J1, H1_F1J2, H1_F1J3,
                        H1_F2J1, H1_F2J2, H1_F2J3,
                        H1_F3J1, H1_F3J2, H1_F3J3):
        robustness = []
        for _ in range(REPEAT_GRASP):
            robustness.append(self.experiment(grasp_distance,
                                              H1_F1J1, H1_F1J2, H1_F1J3,
                                              H1_F2J1, H1_F2J2, H1_F2J3,
                                              H1_F3J1, H1_F3J2, H1_F3J3))

        # trying to maximize the robustness average - while minimizing its variance
        utility = mean(robustness) / max(0.001,sqrt(var(robustness))) # don't divide by 0

        return utility

    def experiment(self, grasp_distance,
                   H1_F1J1, H1_F1J2, H1_F1J3,
                   H1_F2J1, H1_F2J2, H1_F2J3,
                   H1_F3J1, H1_F3J2, H1_F3J3):
        self.sgs.reset_world()
        time.sleep(0.1)
        self.sgs.reset_world()
        time.sleep(0.1)

        self.sgs.open_hand()
        time.sleep(0.1)
        self.sgs.open_hand()
        time.sleep(0.01)

        ball_pose = self.sgs.get_object_pose()
        ball_pose.position.z += 0.5

        #setting an absolute orientation (from the top)
        quaternion = quaternion_from_euler(-pi/2., 0.0, 0.0)
        ball_pose.orientation.x = quaternion[0]
        ball_pose.orientation.y = quaternion[1]
        ball_pose.orientation.z = quaternion[2]
        ball_pose.orientation.w = quaternion[3]

        self.sgs.move_tip_absolute(ball_pose)

        self.sgs.move_tip(y=grasp_distance)

        # close the grasp
        self.sgs.check_fingers_collisions(False)

        self.current_grasp["H1_F1J1"] = H1_F1J1
        self.current_grasp["H1_F1J2"] = H1_F1J2
        self.current_grasp["H1_F1J3"] = H1_F1J3

        self.current_grasp["H1_F2J1"] = H1_F2J1
        self.current_grasp["H1_F2J2"] = H1_F2J2
        self.current_grasp["H1_F2J3"] = H1_F2J3

        self.current_grasp["H1_F3J1"] = H1_F3J1
        self.current_grasp["H1_F3J2"] = H1_F3J2
        self.current_grasp["H1_F3J3"] = H1_F3J3

        self.sgs.send_command(self.current_grasp, duration=0.5)

        # lift slowly and check the quality
        joint_names = self.current_grasp.keys()

        robustness = self.check_stable(joint_names)

        rospy.loginfo("Grasp quality = " + str(robustness))

        sgs.check_fingers_collisions(True)
        # reward
        return robustness

# States
# Assumptions:
# There is no noise in kinematic control
# Perfect localization of objects
# States variables : finger positions, grasp distance,

# Position of the ball, position of the robot arm, distance to ball, finger positions

# Start with random finger joint values with regard to shadow robotic hand

# Start with slightly random arm position and ball position



grasp_distance = -0.16338
def __compute_euclidean_distance(self):
    ball_pose = self.sgs.get_object_pose()
    hand_pose = self.sgs.get_tip_pose()
    dist = sqrt((hand_pose.position.x - ball_pose.position.x)**2 + \
                (hand_pose.position.y - ball_pose.position.y)**2 + \
                (hand_pose.position.z - ball_pose.position.z)**2)
    return dist




def initializeAnExperiment():
    # Reset the world
    sgs.reset_world()
    # Give gazebo some time
    time.sleep(0.1)
    # Double reset vs Gazebo
    sgs.reset_world()
    # Give gazebo some time
    time.sleep(0.1)
    sgs.open_hand()
    time.sleep(0.1)
    sgs.open_hand()
    time.sleep(0.1)
    state = []
    init_joint_state = sgs.get_current_joint_state()
    init_grasp_distance_state_x = 0.0
    init_grasp_distance_state_y = 0.0
    init_grasp_distance_state_z = 0.0

    H1_F1J1  =  (np.random.rand() / 4 )  + 0.6
    H1_F1J2  =  (np.random.rand() / 4 )  + 0.4
    H1_F1J3  =  (np.random.rand() / 4 )  + 0.4
    H1_F2J1  =  (np.random.rand() / 4 )  - 0.1
    H1_F2J2  =  (np.random.rand() / 4 )  + 0.1
    H1_F2J3  =  (np.random.rand() / 4 )  + 0.1
    H1_F2J3  =  (np.random.rand() / 4 )  + 0.0
    H1_F3J1  =  (np.random.rand() / 4 )  - 0.1
    H1_F3J2  =  (np.random.rand() / 4 )  + 0.1
    H1_F3J3  =  (np.random.rand() / 4 )  + 0.4


    state.append(H1_F1J1)
    state.append(H1_F1J2)
    state.append(H1_F1J3)
    state.append(H1_F2J1)
    state.append(H1_F2J2)
    state.append(H1_F2J3)
    state.append(H1_F3J1)
    state.append(H1_F3J2)
    state.append(H1_F3J3)
    state.append(0)

    random_distort_of_hand_x = np.random.rand()  / 5.0 - 0.2
    random_distort_of_hand_z = np.random.rand() / 5.0 - 0.2
    sgs.move_tip(x = random_distort_of_hand_x, z = random_distort_of_hand_z)
    time.sleep(0.1)

    ball_pose = sgs.get_object_pose()
    ball_pose.position.z += 0.5

    #setting an absolute orientation (from the top)
    quaternion = quaternion_from_euler(-pi/2., 0.0, 0.0)
    ball_pose.orientation.x = quaternion[0]
    ball_pose.orientation.y = quaternion[1]
    ball_pose.orientation.z = quaternion[2]
    ball_pose.orientation.w = quaternion[3]


    return sgs, state

def take_step(current_state, action):
    next_state = current_state[0].tolist()
    reward = 0

    experiment_cont = True

    if action < 18:
        # Joint movement positive or negative
        if action % 2 == 0:
            next_state[action / 2] -=  JOINT_STEP_SIZE

        if action % 2 == 1:

            next_state[action / 2] +=  JOINT_STEP_SIZE


    else:
        # Arm movement
        reward = grasp_quality.run_experiments(grasp_distance, next_state[0], next_state[1], next_state[2], next_state[3]
                                , next_state[4], next_state[5], next_state[6], next_state[7], next_state[8])
        if reward > 10:
            experiment_cont = False
    next_state = np.asarray(next_state, dtype=np.float32).reshape((1, 10))
    return next_state, reward, experiment_cont

memory = []
discount = 0.99
max_memory = 500

def __init__(max_memory=500, discount=.99):
    max_memory = max_memory
    memory = list()
    discount = discount

def remember(states, game_over):
    memory.append([states, game_over])
    if len(memory) > max_memory:
        del memory[0]

def get_batch(model,  batch_size=10):
    len_memory = len(memory)
    num_actions = model.output_shape[-1]

    # env_dim = self.memory[0][0][0].shape[1]
    env_dim = 10
    inputs = np.zeros((min(len_memory, batch_size), env_dim))
    targets = np.zeros((inputs.shape[0], num_actions))
    for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
        state_t, action_t, reward_t, state_tp1 = memory[idx][0]
        game_over = memory[idx][1]

        inputs[i:i+1] = state_t
        # There should be no target values for actions not taken.
        # Thou shalt not correct actions not taken #deep
        targets[i] = model.predict(state_t)[0]
        #from IPython.core.debugger import Tracer; Tracer()()

        action_index = np.argmax(model.predict(state_tp1)[0])
        value = model.predict(state_tp1)[0][action_index]
        Q_sa = value
        if game_over:  # if game_over is True

            targets[i, action_t] = reward_t
        else:
            # reward_t + gamma * max_a' Q(s', a')
            targets[i, action_t] = reward_t + discount * Q_sa
        return inputs, targets


# Number of episodes
NUM_OF_EPISODES = 20
# Max steps
NUM_MAX_STEPS = 1000
STATE_SIZE = 10
# Increase, decrease, doing nothing doesn't change the system
# Creates 3 * Degrees of Freedom of the whole system + Grasp or not at that state
NUM_OF_JOINTS = 1
NUM_OF_ACTIONS =  2 * 9   + 1  # 19 number of actions
# Move a joint by a constant value
JOINT_STEP_SIZE = 0.01
i_episode = 0
hidden_size = 50
learning_rate = 0.99
discount_rate = 0.9
batch_size = 50
epsilon = 0.3
sgs, state = initializeAnExperiment()
# Use an old state, if need to continue with training
#f = open('state.pckl', 'rb')
#state = pickle.load(f)
#f.close()
experiment_cont = True
grasp_quality = GraspQuality(sgs)
epsilonGrasp = 0.05
# Use a saved model if you want
model = keras.models.load_model('model-Q.h5')

#model = Sequential()
#model.add(Dense(hidden_size, input_shape=(STATE_SIZE, ), activation='relu'))
#model.add(Dense(hidden_size, activation='relu'))
#model.add(Dense(hidden_size, activation='relu'))
#model.add(Dense(NUM_OF_ACTIONS))
#model.compile(Adam(lr=0.01), "mse")




def check_stable(self, joint_names):
    current_min = 1000
    positions = []
    velocities = []
    efforts = []
    for k in range(30):
        sgs.move_tip(y=0.02)
        ball_distance = self.__compute_euclidean_distance()
        if k > MIN_LIFT_STEPS and ball_distance < current_min:
            current_min = ball_distance
            break
        if ball_distance > MAX_BALL_DISTANCE:
            break

        time.sleep(0.5)
    reward = (1/(current_min - 0.18))**2
    return reward



for i_episode in range(NUM_OF_EPISODES):
    step = 0
    loss = 0
    current_state = np.asarray(state, dtype=np.float32).reshape((1, 10))
    print 'Episode:' + str(i_episode)
    experiment_cont = True
    while experiment_cont:
        step += 1

        # Convert list to keras friendly numpy shape

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, NUM_OF_ACTIONS - 1, size=1)[0]
        else:
            q = model.predict(current_state)
            action = np.argmax(q[0]) - 1

        if np.random.rand() <= epsilonGrasp:
            action = 18


        #print action
        next_state, reward, experiment_cont = take_step(current_state, action)
        remember([next_state,  action, reward, current_state], (not experiment_cont))

        inputs, targets = get_batch(model, batch_size=batch_size)

        loss += model.train_on_batch(inputs, targets) #[0]
        current_state = next_state
        if step > NUM_MAX_STEPS:
            next_state, reward, experiment_cont = take_step(current_state, 18)
            inputs, targets = get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets) #[0]
            break
    print 'Total Loss:' + str(loss)
    print 'Total number of steps:' + str(step)
        #actions = policy.getActions(current_state)
        #next_state, experiment_cont = step(sgs, action)
