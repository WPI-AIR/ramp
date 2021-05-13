#!/usr/bin/env python

import os
import sys
import rospy
import numpy as np
import gym
from gym.spaces import prng
import datetime
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.optimizers import Adam

ramp_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(ramp_root) # directory_name

import sys
sys.path.append("~/catkin_ws/src/ramp/keras_rl/dist/keras_rl-0.4.0-py2.7.egg")
# import rl

from rl.agents.dqn_si import DQNAgentSi
from rl.policy import BoltzmannQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from ramp_rlpara.ramp_gym.ramp_env_interfaces.ramp_env_interface_sipd import RampEnvSipd
rospy.init_node('dqn_ramp_sipd', anonymous=True)

## make directory of logging
home_dir = os.getenv("HOME")
cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
file_dir = home_dir + '/data/ramp/ramp_rlpara/dqn_ramp_sipd/' + cur_date + '/raw_data/'
os.system('mkdir -p ' + file_dir)

from f_logger import RampRlLogger
## Initialize logger
coarse_logger = RampRlLogger(file_dir + "dqn_sipd.csv",
                             ['plan#', 'Ap', 'Bp', 'L','F','dp_obs', 'min_obs_dis',
                              'plan_reward', 'plan_time', 'obs_dis',
                              'loss', 'mae', 'mean_q'])

epi_logger = RampRlLogger(file_dir + "dqn_epi_sipd.csv",
                             ['epi#',
                              'epi_reward',
                              'epi_steps'])



# Seed
# seed_str = input("Enter a seed for random generator (must be a integer): ")
# seed_int = int(seed_str)
seed_int = 0
np.random.seed(seed_int) # numpy
prng.seed(seed_int) # space



# Get the environment and extract the number of actions.
env = RampEnvSipd('ramp_sipd')
nb_actions = env.action_space.n


# Test
# ob, coes = env.reset()
# print(ob)
# print(coes)

# while not rospy.core.is_shutdown():
#     ob, r, d, info = env.step(4)
#     print(ob)
#     print(r)
#     print(d)
#     print(info)



# Next, we build a very simple model Q(s,a).
model = Sequential()
model.add(LSTM(16, input_shape=(1,10)))
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # s is (x, y, coe)
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions)) # Q values, number is nb_actions
model.add(Activation('linear'))
print(model.summary())



init_boltz_tau = 1.0



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100, window_length=1)
policy = BoltzmannQPolicy(tau=init_boltz_tau)
dqn = DQNAgentSi(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=300,
               target_model_update=0.001, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])



# Load weights if needed. Put this after compiling may be better.
# dqn.load_weights_sip("dqn_ramp_sipd_weights.h5f")
dqn.model.load_weights("/home/sapanostic/catkin_ws/src/ramp/ramp_rlpara/scripts/1m/raw_data/dqn_ramp_sipd_weights_model.h5f")
dqn.target_model.load_weights("/home/sapanostic/catkin_ws/src/ramp/ramp_rlpara/scripts/1m/raw_data/dqn_ramp_sipd_weights_target.h5f")


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
log_interval = 1000
nb_max_episode_steps = None
# dqn.fitSip(env, nb_steps=5000, log_interval=log_interval,
#            nb_max_episode_steps=nb_max_episode_steps, verbose=2,
#            file_dir=file_dir, logger=coarse_logger, epi_logger=epi_logger)

# dqn.fitSip(env, nb_steps=5000000, log_interval=log_interval,
#            nb_max_episode_steps=nb_max_episode_steps, verbose=0,
#            file_dir=file_dir)

# # After training is done, we save the final weights.
dqn.save_weights_sip(file_dir + 'dqn_{}_weights.h5f'.format(env.name), overwrite = True)
# coarse_logger.close()



# # Finally, evaluate our algorithm for 5 episodes.
dqn.testSip(env, nb_episodes=10, visualize=False, nb_max_episode_steps=3000)

plt.show()

while(1):
	print("waiting for user to close the program")