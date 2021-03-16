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
import math
from colorama import init as clr_ama_init
from colorama import Fore
import itertools
clr_ama_init(autoreset = True)

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



    def __init__(self, name='ramp_sipd'):
        self.name = name
        self.check_best_rate = rospy.Rate(10) # 0.1s
        
        ## get various parameters
        self.utility = Utility()
        self.action_resolution = 0.05
        self.done = False
        self.start_in_step = False
        self.max_reward = -999.9

        self.a0 = 0.0
        self.a1 = 1.0
        self.b0 = 0.0
        self.b1 = 1.0
        self.d0 = 0.0
        self.d1 = 1.0
        self.l0 = 0.0
        self.l1 = 1.0
        self.k0 = 0.0
        self.k1 = 1.0
        #self.best_A = 0.10
        #self.best_D = 0.15

        # what values for best?

        self.best_Ap = 0.10
        self.best_Bp = 0.10
        self.best_dp = 0.10
        self.best_L = 0.10
        self.best_k = 0.10
        #self.preset_A = 0.05
        #self.preset_D = 0.65
        
        self.action_space = spaces.Discrete(243) #81
        #self.observation_space = spaces.Box(np.array([self.utility.min_x, self.utility.min_y, self.a0, self.b0, self.d0. self.l0, self.k0]),
        #                                    np.array([self.utility.max_x, self.utility.max_y, self.a1, self.b1, self.d1. self.l1, self.k1])) # single motion state
        
        self.observation_space = spaces.Box(np.array([self.utility.min_x, self.utility.min_y, self.a0, self.b0, self.d0, self.l0, self.k0]),
                                            np.array([self.utility.max_x, self.utility.max_y, self.a1, self.b1, self.d1, self.l1, self.k1])) # single motion state
        self.best_traj = None
        self.best_t = None
        self.this_exe_info = None

        self.best_traj_sub = rospy.Subscriber("bestTrajec", RampTrajectory,
                                                 self.bestTrajCallback)

        self.one_exe_info_sub = rospy.Subscriber("ramp_collection_ramp_ob_one_run", RampObservationOneRunning,
                                                 self.oneExeInfoCallback)

        self.setState(1.0, 0.45, 0.80, 0.50, 0.50) # TODO: check state values (hyperparam?)



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
            if has_waited_for >= 20.0: # overtime
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
        dp = rospy.get_param('/ramp/eval_weight_dp')
        L = rospy.get_param('/ramp/eval_weight_L')
        k = rospy.get_param('/ramp/eval_weight_k')
        coes = np.array([Ap, Bp, dp, L, k])

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
        action_space_matrix = list(set(itertools.permutations([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2], 5)))

        dAp = action_space_matrix[action][0]
        dBp = action_space_matrix[action][1]
        ddp = action_space_matrix[action][2]
        dL = action_space_matrix[action][3]
        dk = action_space_matrix[action][4]

        dAp = (dAp - 1) * self.action_resolution
        dBp = (dBp - 1) * self.action_resolution
        ddp = (ddp - 1) * self.action_resolution
        dL = (dL - 1) * self.action_resolution        
        dk = (dk - 1) * self.action_resolution         

        return dAp, dBp, ddp, dL, dk



    def step(self, action):
        print('################################################################')
        dAp, dBp, ddp, dL, dk = self.decodeAction(action)

        ## set the coefficients of RAMP
        Ap = rospy.get_param('/ramp/eval_weight_Ap')
        Bp = rospy.get_param('/ramp/eval_weight_Bp')
        dp = rospy.get_param('/ramp/eval_weight_dp')
        L = rospy.get_param('/ramp/eval_weight_L')
        k = rospy.get_param('/ramp/eval_weight_k')
        self.setState(Ap+dAp, Dp+dDp, dp+ddp, L+dL, k+dk)

        self.oneCycle(start_planner=self.start_in_step)
        # Reward are for the whole path and its coefficients.
        return self.getOb(), self.getReward(), self.done, self.getInfo()



    def setState(self, Ap, Bp, dp, L, k):
        Ap = np.clip(Ap, self.a0, self.a1)
        Bp = np.clip(Bp, self.b0, self.b1)
        dp = np.clip(dp, self.d0, self.d1)
        L = np.clip(L, self.l0, self.l1)
        k = np.clip(k, self.k0, self.k1)
        rospy.set_param('/ramp/eval_weight_Ap', Ap.item())
        rospy.set_param('/ramp/eval_weight_Bp', Bp.item())
        rospy.set_param('/ramp/eval_weight_dp', dp.item())
        rospy.set_param('/ramp/eval_weight_L', L.item())
        rospy.set_param('/ramp/eval_weight_k', k.item())



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
        if self.best_t is None:
            reward = 0.0
        else:
            orien_cost = self.best_t.orien_fitness

            if self.best_t.obs_fitness < 0.1:
                obs_cost = 1.25
            else:
                obs_cost = 1.0 / self.best_t.obs_fitness

            reward = -obs_cost / 5.0

        if reward > self.max_reward:
            self.max_reward = reward
            self.best_Ap = rospy.get_param('/ramp/eval_weight_Ap')
            self.best_Bp = rospy.get_param('/ramp/eval_weight_Bp')
            self.best_dp = rospy.get_param('/ramp/eval_weight_dp')
            self.best_L = rospy.get_param('/ramp/eval_weight_L')
            self.best_k = rospy.get_param('/ramp/eval_weight_k')

        if self.done:
            reward += 30.0 # TODO: Remove this?

        return reward



    def getOb(self):
        Ap = rospy.get_param('/ramp/eval_weight_Ap')
        Bp = rospy.get_param('/ramp/eval_weight_Bp')
        dp = rospy.get_param('/ramp/eval_weight_dp')
        L = rospy.get_param('/ramp/eval_weight_L')
        k = rospy.get_param('/ramp/eval_weight_k')

        if self.best_t is None or len(self.best_t.holonomic_path.points) < 2:
            return np.array([[0.0, 0.0, Ap, Bp, dp, L, k]])

        x1 = self.best_t.holonomic_path.points[1].motionState.positions[0]
        y1 = self.best_t.holonomic_path.points[1].motionState.positions[1]
        ob = np.array([[x1, y1, Ap, Bp, dp, L, k]])
        length = len(self.best_t.holonomic_path.points)
        for i in range(2, length):
            xi = self.best_t.holonomic_path.points[i].motionState.positions[0]
            yi = self.best_t.holonomic_path.points[i].motionState.positions[1]
            ob = np.concatenate((ob, [[xi, yi, Ap, Bp, dp, L, k]]))

        return ob



    def getInfo(self):
        return {'time': 0.0,
                'obs_dis': 0.0}



    def getState(self):
        return 0.0