'''
This is not the environment itself but the interface of environment.
Observation is a single motion state whose size is 10
'''

import os
import sys
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import rospy
import warnings
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Empty
from ramp_msgs.msg import RampTrajectory
from ramp_msgs.msg import RampObservationOneRunning

## get directory
rlpara_root = os.path.join(os.path.dirname(__file__), '../../')
lib_dir = os.path.join(rlpara_root, 'lib/')
sys.path.append(lib_dir)

## from .py_file_name import class_name
from f_utility import Utility

class RampEnv(gym.Env):

	def setEnvRdyTrueCallback(self, data):
		self.env_ready = True

	def oneExeInfoCallback(self, data):
		if self.this_exe_info is not None:
			## TODO: log into file with date in its name
			warnings.warn("Unused RampEnv.this_exe_info is overlaped!")
		self.this_exe_info = data

	def __init__(self):
		self._seed()
		
		## get various parameters
		self.utility = Utility()

		a0 = self.utility.coe_A_range[0]
		a1 = self.utility.coe_A_range[1]

		d0 = self.utility.coe_D_range[0]
		d1 = self.utility.coe_D_range[1]

		qk0 = self.utility.coe_Qk_range[0]
		qk1 = self.utility.coe_Qk_range[1]

		t0 = 0.0
		t1 = self.utility.time_stamp_max

		x0 = self.utility.min_x
		x1 = self.utility.max_x

		y0 = self.utility.min_y
		y1 = self.utility.max_y

		theta0 = self.utility.min_theta
		theta1 = self.utility.max_theta

		x_d0 = -self.utility.max_linear_v
		x_d1 =  self.utility.max_linear_v

		y_d0 = -self.utility.max_linear_v
		y_d1 =  self.utility.max_linear_v

		theta_d0 = -self.utility.max_angular_v
		theta_d1 =  self.utility.max_angular_v

		x_dd0 = -self.utility.max_linear_a
		x_dd1 =  self.utility.max_linear_a

		y_dd0 = -self.utility.max_linear_a
		y_dd1 =  self.utility.max_linear_a

		theta_dd0 = -self.utility.max_angular_a
		theta_dd1 =  self.utility.max_angular_a
		
		self.action_space = spaces.Box(np.array([a0, d0, qk0]),
		                               np.array([a1, d1, qk1]))
		self.observation_space = spaces.Box(np.array([t0, x0, y0, theta0, x_d0, y_d0, theta_d0, x_dd0, y_dd0, theta_dd0]),
		                                    np.array([t1, x1, y1, theta1, x_d1, y_d1, theta_d1, x_dd1, y_dd1, theta_dd1]))
		self.env_ready = False
		self.this_exe_info = None

		self.set_env_rdy_true_sub = rospy.Subscriber("set_env_ready_true", Empty, self.setEnvRdyTrueCallback)
		self.one_exe_info_sub = rospy.Subscriber("ramp_collection_ramp_ob_one_run", RampObservationOneRunning, self.oneExeInfoCallback)

		self.si_act_pub = rospy.Publisher('ramp_collection_si_act', Float64MultiArray, queue_size = 1)

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def _reset(self):
		## not used in small input version
		pass

	def _step(self, action):
		assert self.action_space.contains(action)

		## set the coefficients of RAMP
		rospy.set_param("/ramp/eval_weight_A", action[0].item())
		rospy.set_param("/ramp/eval_weight_D", action[1].item())
		rospy.set_param("/ramp/eval_weight_Qk", action[2].item())

		## wait the actual environment to get ready......
		print("wait the actual environment to get ready......")
		while not self.env_ready:
			time.sleep(0.02) # 0.02s
		print("find env. ready and set start_planner True for the ready env. to start one execution!")
		rospy.set_param("/ramp/start_planner", True)
		self.env_ready = False

		## here you can publish sth. to "/ramp_collection_.*"
		self.si_act_pub.publish(Float64MultiArray(data = action.tolist()))
		
		## wait for this execution completes......
		print("wait for this execution completes......")
		while self.this_exe_info is None: # TODO: enable key interrupt
			time.sleep(0.1) # 0.1s
		print("A execution completes!")
		observations = self.this_exe_info # build many observations used for returning
		self.this_exe_info = None # clear self.this_exe_info after it is used

		## calculate reward
		reward = self.utility.max_exe_time - observations.execution_time
		reward = max(0, reward)

		## done or not
		done = observations.done

		## reward and done are both for the last observation in this execution
		return observations, reward, done, {}
