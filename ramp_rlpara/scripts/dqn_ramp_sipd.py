#!/usr/bin/env python

import os
import sys
import rospy
import numpy as np
import gym
from gym.spaces import prng
import datetime
import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf
from tensorflow import keras
import random

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam

ramp_root = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(ramp_root) # directory_name

import sys
sys.path.append("~/catkin_ws/src/ramp/keras_rl/dist/keras_rl-0.4.0-py2.7.egg")
# import rl

# from rl.agents.dqn_si import DQNAgentSi
from rl.policy import BoltzmannQPolicy
# from rl.policy import GreedyQPolicy
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

from ramp_rlpara.ramp_gym.ramp_env_interfaces.ramp_env_interface_sipd import RampEnvSipd
rospy.init_node('dqn_ramp_sipd', anonymous=True)

## make directory of logging
home_dir = os.getenv("HOME")
cur_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
file_dir = home_dir + '/data/ramp/ramp_rlpara/dqn_ramp_sipd/' + cur_date + '/raw_data/'
os.system('mkdir -p ' + file_dir)

from f_logger import RampRlLogger

# def create_model(env, nb_actions):
# 	model = Sequential()
# 	model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # s is (x, y, coe)
# 	model.add(Dense(16))
# 	model.add(Activation('relu'))
# 	model.add(Dense(16))
# 	model.add(Activation('relu'))
# 	model.add(Dense(16))
# 	model.add(Activation('relu'))
# 	model.add(Dense(nb_actions)) # Q values, number is nb_actions
# 	model.add(Activation('linear'))
# 	print(model.summary())

# 	return model

## Initialize logger
# coarse_logger = RampRlLogger(file_dir + "dqn_sipd.csv",
#                              ['plan#', 'A', 'D',
#                               'plan_reward', 'plan_time', 'obs_dis',
#                               'loss', 'mae', 'mean_q'])

# epi_logger = RampRlLogger(file_dir + "dqn_epi_sipd.csv",
#                              ['epi#',
#                               'epi_reward',
#                               'epi_steps'])

class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma

        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        
    def build_model(self):
        model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Embedding(input_dim=100, output_dim=64))
        model.add(tf.keras.layers.LSTM(16))
        model.add(keras.layers.Dense(24, activation='relu')) #[Input] -> Layer 1
        model.add(keras.layers.Dense(24, activation='relu')) #[Input] -> Layer 2
        model.add(keras.layers.Dense(24, activation='relu')) #[Input] -> Layer 3
        model.add(keras.layers.Dense(self.nA, activation='linear')) #Layer 4 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        # model.summary()
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA) #Explore
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory 

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch) # array of tuples 
        print(np_array.shape)

        # ToDo: Batch Normalization for Recurrent Networks 
        # st = np.zeros((0,1,self.nS)) #States
        # nst = np.zeros( (0,self.nS) )#Next States
        # print(st.shape)
        # print(np_array[0,0].shape)
        # for i in range(len(np_array)): #Creating the state and next state np arrays
        #     st = np.append( st, np_array[i,0], axis=0)
        #     nst = np.append( nst, np_array[i,3], axis=0)
        
        # Lets go without batch normalization for now. 
        st = np.array(minibatch)[0][0]
        nst = np.array(minibatch)[0][3]
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(np.expand_dims(nst, axis=0))
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        # x_reshape = np.array(x).reshape(batch_size,self.nS)
        # y_reshape = np.array(y)
        x_reshape = state
        y_reshape = np.expand_dims(target_f,axis=0)
        print(x_reshape, x_reshape.shape)
        print(y_reshape, y_reshape.shape)
        epoch_count = 1 #Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, batch_size=batch_size, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Seed
# seed_str = input("Enter a seed for random generator (must be a integer): ")
seed_str = 1
seed_int = int(seed_str)
np.random.seed(seed_int) # numpy
prng.seed(seed_int) # space


# Get the environment and extract the number of actions.
env = RampEnvSipd('ramp_sipd')

# Test
ob, coes = env.reset()
nA = env.action_space.n
nS = ob.shape[1]
# print("nS= ", nS, " nA= ", nA)
# print("Testing Enviornment")
# print("observation:", ob)
# print("coes:", coes)

## Parameters 
EPISODES = 10
learning_rate = 0.95    # alpha
discount_rate = 0.001   # Gamma 
batch_size = 1         # 
epsilon = 1
epsilon_min = 0.001
epsilon_decay = 0.995      
init_boltz_tau = 1.0

dqn = DeepQNetwork(nS, nA, learning_rate, discount_rate, epsilon, epsilon_min, epsilon_decay)
policy = BoltzmannQPolicy(tau=init_boltz_tau)

observation = env.reset()
tf.compat.v1.disable_eager_execution()
# dqn.model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
#                       optimizer=keras.optimizers.Adam(lr=learning_rate)) #Optimaizer: Adam (Feel free to check other options)
# dqn.model.fit( epochs=1)
####### Training ########
rewards = []
epsilons = []
done = False 

for e in range(EPISODES):
  observation, coe = env.reset()
  total_rewards = 0 # Episodic Reward

  while not done: # ToDo: What are End of Episode conditions? 
    print("------------------------------------------------------- ")
    # action = dqn.action(observation)
    observation = np.expand_dims(observation, axis=0)
    q_values = dqn.model.predict(observation) # 
    action = policy.select_action(q_values[0])
    print("Episode: ", e, " \n observation:", observation, "\n action: ", action)
    next_observation, reward, done, info = env.step(action)
    total_rewards += reward
    dqn.store(observation, action, reward, next_observation, done)
    observation = next_observation
    break

  rewards.append(total_rewards)
  epsilons.append(dqn.epsilon)
  print("episode: {}/{}, score: {}, e: {}"
                  .format(e, EPISODES, total_rewards, dqn.epsilon))

  if len(dqn.memory) > batch_size:
            dqn.experience_replay(batch_size)


# Next, we build a very simple model Q(s,a).
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # s is (x, y, coe)
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions)) # Q values, number is nb_actions
# model.add(Activation('linear'))
# print(model.summary())
# model = create_model(env, nb_actions)

# init_boltz_tau = 1.0



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
# memory = SequentialMemory(limit=100, window_length=1)
# policy = BoltzmannQPolicy(tau=init_boltz_tau)
# dqn = DQNAgentSi(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=300,
#                target_model_update=0.001, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])



# Load weights if needed. Put this after compiling may be better.
# dqn.load_weights_sip("~/catkin_ws/data/ramp/ramp_rlpara/dqn_ramp_sipd/2018-02-17_14:05:42/raw_data/58/" +
#                      "dqn_{}_weights.h5f".format(env.name))



# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# log_interval = 1000
# nb_max_episode_steps = None
# dqn.fitSip(env, nb_steps=5000000, log_interval=log_interval,
#            nb_max_episode_steps=nb_max_episode_steps, verbose=2,
#            file_dir=file_dir, logger=coarse_logger, epi_logger=epi_logger)



# # After training is done, we save the final weights.
# dqn.save_weights_sip(file_dir + 'dqn_{}_weights.h5f'.format(env.name), overwrite = True)
# coarse_logger.close()



# # Finally, evaluate our algorithm for 5 episodes.
# dqn.testSip(env, nb_episodes=11, visualize=False, nb_max_episode_steps=3000)

# plt.show()