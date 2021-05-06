'''
This is not the environment itself but the interface of environment.
Observation is a path and its corresponding coefficients.
'''

import os
import sys
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Empty
from ramp_msgs.msg import RampTrajectory
from ramp_msgs.msg import RampObservationOneRunning
from pedsim_msgs.msg import AgentStates
import math
from colorama import init as clr_ama_init
from colorama import Fore
clr_ama_init(autoreset = True)
import itertools

import tensorflow as tf 
import datetime

## get directory
rlpara_root = os.path.join(os.path.dirname(__file__), '../../')
lib_dir = os.path.join(rlpara_root, 'lib/')
sys.path.append(lib_dir)

## from .py_file_name import class_name
from f_utility import Utility

class RampEnvSipd(gym.Env):



    def bestTrajCallback(self, data):
        if data.fitness < -1000:
            return

        if self.best_traj is not None:
            print(Fore.RED + "Unused best_traj is overlaped!")
        self.best_traj = data



    def oneExeInfoCallback(self, data):
        if self.this_exe_info is not None:
            print(Fore.RED + "Unused this_exe_info is overlaped!")
        self.this_exe_info = data

    def pesimCallback(self, data):
        fx = data.agent_states[0].forces.social_force.x    
        fy = data.agent_states[0].forces.social_force.y
        self.pedforce = np.sqrt(fx*fx + fy*fy)
        

    def __init__(self, name='ramp_sipd'):
        self.name = name
        self.check_best_rate = rospy.Rate(10) # 0.1s
        
        ## get various parameters
        self.utility = Utility()
        self.action_resolution = 0.05
        self.done = False
        self.start_in_step = False
        self.max_reward = -999.9

        self.a0 = 1.0
        self.a1 = 2.0
        self.b0 = 0.1
        self.b1 = 1.0
        self.l0 = 0.0
        self.l1 = 1.0
        self.best_Ap = 0.10
        self.best_Bp = 0.10
        self.best_L = 0.10
        self.action_space = spaces.Discrete(27) # 3 * 3 = 9
        self.observation_space = spaces.Box(np.array([self.utility.min_x, self.utility.min_y, self.utility.min_theta, self.utility.min_linear_vx, self.utility.min_linear_vy, self.utility.min_angular_v, self.utility.min_linear_ax, self.utility.min_linear_ay, self.utility.min_angular_a, self.utility.min_time]),
                                            np.array([self.utility.max_x, self.utility.max_y, self.utility.max_theta, self.utility.max_linear_vx, self.utility.max_linear_vy, self.utility.max_angular_v, self.utility.max_linear_ax, self.utility.max_linear_ay, self.utility.max_angular_a, self.utility.max_time])) # single motion state
        self.best_traj = None
        self.best_t = None
        self.this_exe_info = None
        self.pedforce = 0
        self.best_traj_sub = rospy.Subscriber("bestTrajec", RampTrajectory,
                                                 self.bestTrajCallback)

        self.one_exe_info_sub = rospy.Subscriber("ramp_collection_ramp_ob_one_run", RampObservationOneRunning,
                                                 self.oneExeInfoCallback)

        self.pedsim_sub = rospy.Subscriber("pedsim_simulator/simulated_agents",  AgentStates,
                                                 self.pesimCallback)

        self.setState(1.0, 1.0, 1.0) # TODO: check state values (hyperparam?)

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.session = tf.compat.v1.InteractiveSession(config=config)
        if not os.path.exists('summaries'):
            print("Creating summaries folder ...")
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))

        self.summ_writer = tf.compat.v1.summary.FileWriter("/home/sapanostic/data/runs")
        self.step_i = 0

    def oneCycle(self, start_planner=False):
        """Wait for ready, start one execution and wait for its completion.

        Returns
        -------
            Whether this cycle succeeds or not.
        """
        if start_planner:
            ## Wait environment get ready......
            print("Wait environment get ready......")
            try:
                rospy.wait_for_service("env_ready_srv")
            except rospy.exceptions.ROSInterruptException:
                print("\nCtrl+C is pressed!")
                return False

            print("Start one execution!")
            rospy.set_param("/ramp/start_planner", True)
        
        ## Wait plan complete......
        has_waited_exe_for = 0
        print("Wait plan complete......")
        start_waiting_time = rospy.get_rostime()
        while not rospy.core.is_shutdown() and self.best_traj is None and self.this_exe_info is None:
            try:
                self.check_best_rate.sleep()
            except rospy.exceptions.ROSInterruptException:
                print("\nCtrl+C is pressed!")
                return False

            cur_time = rospy.get_rostime()
            has_waited_for = cur_time.to_sec() - start_waiting_time.to_sec()
            if has_waited_for >= 40.0: # overtime
                print("Long time no response!")
                print("Wait environment get ready......")

                try:
                    rospy.wait_for_service("env_ready_srv")
                except rospy.exceptions.ROSInterruptException:
                    print("\nCtrl+C is pressed!")
                    return False

                print("Start one execution!")
                rospy.set_param("/ramp/start_planner", True)
                start_waiting_time = rospy.get_rostime()
                print("Wait plan complete......")

        print("A plan completes!")
        time.sleep(0.1)
        self.best_t = self.best_traj # Store
        if self.this_exe_info is not None:
            self.done = self.this_exe_info.done
            self.start_in_step = (not self.this_exe_info.done)
        else:
            self.done = False
            self.start_in_step = False
        self.best_traj = None # Clear self.best_traj after it is used
        self.this_exe_info = None

        return True



    def reset(self, full_rand=True):
        """Set two coefficients and do one plan.

        Returns
        -------
            Multiple single states.
        """
        # coes = np.array([self.best_A, self.best_D])
        # coes = np.random.rand(2)
        # self.setState(coes[0], coes[1])

        Ap = rospy.get_param('/ramp/eval_weight_Ap')
        Bp = rospy.get_param('/ramp/eval_weight_Bp')
        L = rospy.get_param('/ramp/eval_weight_L')
        coes = np.array([Ap, Bp, L])

        self.oneCycle(start_planner=True)
        return self.getOb(), coes



    def decodeAction(self, action):
        """
        Arguments
        ---------
            action (int): encoded delta A, D weight.

        Return
        ------
            (float): Delta A, D weight.
        """
        action_space_matrix = list(set(itertools.permutations([0,0,0,1,1,1,2,2,2], 3)))
        dAp = action_space_matrix[action][0]
        dBp = action_space_matrix[action][1]
        dL = action_space_matrix[action][2]

        dAp = (dAp - 1) * self.action_resolution
        dBp = (dBp - 1) * self.action_resolution
        dL = (dL - 1) * self.action_resolution               

        return dAp, dBp, dL



    def step(self, action):
        print('################################################################')
        dAp, dBp, dL = self.decodeAction(action)

        ## set the coefficients of RAMP
        Ap = rospy.get_param('/ramp/eval_weight_Ap')
        Bp = rospy.get_param('/ramp/eval_weight_Bp')
        L = rospy.get_param('/ramp/eval_weight_L')

        if ((Ap+dAp)<=0):
            Ap = dAp
        if ((Bp+dBp)<=0):
            Bp = dBp    
        
        self.setState(Ap+dAp, Bp+dBp, L+dL)
        self.step_i += 1
        
        self.oneCycle(start_planner=self.start_in_step)
        # Reward are for the whole path and its coefficients.
        return self.getOb(), self.getReward(), self.done, self.getInfo()



    def setState(self, Ap, Bp, L):
        Ap = np.clip(Ap, self.a0, self.a1)
        Bp = np.clip(Bp, self.b0, self.b1)
        L = np.clip(L, self.l0, self.l1)
        rospy.set_param('/ramp/eval_weight_Ap', Ap.item())
        rospy.set_param('/ramp/eval_weight_Bp', Bp.item())
        rospy.set_param('/ramp/eval_weight_L', L.item())


    def getReward(self):
        # First
        # A = rospy.get_param('/ramp/eval_weight_A')
        # D = rospy.get_param('/ramp/eval_weight_D')
        # dis = 0.1
        # dis += abs(A - self.preset_A) + abs(D - self.preset_D)
        # dis *= 3.0
        # reward = -dis

        # Second
        # reward = -1.0

        # Third
        reward = 0.0
        obs_cost = 0
        if self.best_t is None:
            reward = 0.0
        else:
            dp_obs = rospy.get_param('/ramp/dp_obs')
            if self.best_t.min_obs_dis < 2.0:
                obs_cost = -10*np.exp(2.0 - self.best_t.min_obs_dis) # order of -27 
                reward += obs_cost

            reward += 10-0.2*self.best_t.time_fitness 
            print("obs: ", self.best_t.min_obs_dis, "time: ", -self.best_t.time_fitness)
            rospy.set_param('/ramp/min_obs_dis', self.best_t.min_obs_dis)
            # print("fitness: ", self.best_t.fitness)
            # print("min_obs_dis: ", self.best_t.min_obs_dis)
            # print("time_fitness: ", self.best_t.time_fitness)
            # print("orien_fitness: ", self.best_t.orien_fitness)
            # print("obs_fitness: ", self.best_t.obs_fitness)
            # print("t_start.secs: ", self.best_t.t_start.secs)
        #     orien_cost = self.best_t.orien_fitness

        #     if self.best_t.obs_fitness < 0.1:
        #         obs_cost = 1.25
        #     else:
        #         obs_cost = 1.0 / self.best_t.obs_fitness

        #     reward = -obs_cost / 5.0

        # if reward > self.max_reward:
        #     self.max_reward = reward

        if self.done:
            print("Reached Goal, Getting BIG REWARD")
            reward += 100.0 # TODO: Remove this?

        return reward



    def getOb(self):
        self.best_Ap = rospy.get_param('/ramp/eval_weight_Ap')
        self.best_Bp = rospy.get_param('/ramp/eval_weight_Bp')
        self.best_L = rospy.get_param('/ramp/eval_weight_L')

        if self.best_t is None or len(self.best_t.holonomic_path.points) < 2:
            return np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        x1 = self.best_t.holonomic_path.points[1].motionState.positions[0]
        y1 = self.best_t.holonomic_path.points[1].motionState.positions[1]
        th1 = self.best_t.holonomic_path.points[1].motionState.positions[2]
        if self.best_t.holonomic_path.points[1].motionState.velocities is () or self.best_t.holonomic_path.points[1].motionState.accelerations is ():
            vx1 = 0.0
            vy1 = 0.0
            vth1 = 0.0
            ax1 = 0.0
            ay1 = 0.0
            ath1 = 0.0
        else:
            vx1 = self.best_t.holonomic_path.points[1].motionState.velocities[0]
            vy1 = self.best_t.holonomic_path.points[1].motionState.velocities[1]
            vth1 = self.best_t.holonomic_path.points[1].motionState.velocities[2]
            ax1 = self.best_t.holonomic_path.points[1].motionState.accelerations[0]
            ay1 = self.best_t.holonomic_path.points[1].motionState.accelerations[1]
            ath1 = self.best_t.holonomic_path.points[1].motionState.accelerations[2]

        time = self.best_t.holonomic_path.points[1].motionState.time

        ob = np.array([[x1, y1, th1, vx1, vy1, vth1, ax1, ay1, ath1, time]])
        length = len(self.best_t.holonomic_path.points)
        for i in range(2, length):
            xi = self.best_t.holonomic_path.points[i].motionState.positions[0]
            yi = self.best_t.holonomic_path.points[i].motionState.positions[1]
            thi = self.best_t.holonomic_path.points[i].motionState.positions[2]

            if self.best_t.holonomic_path.points[i].motionState.velocities is () or self.best_t.holonomic_path.points[i].motionState.accelerations is ():
                vxi = 0.0
                vyi = 0.0
                vthi = 0.0
                axi = 0.0
                ayi = 0.0
                athi = 0.0
            else:
                vxi = self.best_t.holonomic_path.points[i].motionState.velocities[0]
                vyi = self.best_t.holonomic_path.points[i].motionState.velocities[1]
                vthi = self.best_t.holonomic_path.points[i].motionState.velocities[2]
                axi = self.best_t.holonomic_path.points[i].motionState.accelerations[0]
                ayi = self.best_t.holonomic_path.points[i].motionState.accelerations[1]
                athi = self.best_t.holonomic_path.points[i].motionState.accelerations[2]

            time = self.best_t.holonomic_path.points[i].motionState.time
            ob = np.concatenate((ob, [[xi, yi, thi, vxi, vyi, vthi, axi, ayi, athi, time]]))

        return ob



    def getInfo(self):
        return {'time': 0.0,
                'obs_dis': 0.0}



    def getState(self):
        Ap = rospy.get_param('/ramp/eval_weight_Ap')
        Bp = rospy.get_param('/ramp/eval_weight_Bp')
        L = rospy.get_param('/ramp/eval_weight_L')
        # F = self.pedforce
        F = rospy.get_param('/ramp/Fint')
        dp_obs = rospy.get_param('/ramp/dp_obs')
        min_obs_dis = rospy.get_param('/ramp/min_obs_dis')
        return [Ap, Bp, L, F, dp_obs, min_obs_dis] 

    def setLoggingData(self, reward, coes, loss, mae, q, obs_dis):
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="reward", simple_value=reward)])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Ap", simple_value=coes[0])])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Bp", simple_value=coes[1])])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="L", simple_value=coes[2])])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Loss", simple_value=loss)])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="mae", simple_value=mae)])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="q", simple_value=q)])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="obs_dis", simple_value=obs_dis)])
        self.summ_writer.add_summary(summary, global_step=self.step_i)

        

