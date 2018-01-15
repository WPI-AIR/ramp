"""
Observation
-----------
    Path: 2 points, each is in (x,y) form.

State
-----
    Coefficients: only D~[0,1].

Action
------
    Delta coefficients: dD ~ -0.1, 0.0, +0.1.
"""

import gym
from gym import spaces
import numpy as np

class PathDcoeDiscre(gym.Env):
    """docstring for PathDcoeDiscre.
    """
    def __init__(self, name='pdd'):
        super(PathDcoeDiscre, self).__init__()
        self.name = name
        self.state = 0.0 # D weight
        self.state_min = 0.0
        self.state_max = 1.0
        self.max_len = 60.0
        self.goal = np.array([10, 10])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 10, shape=(2,)) # single motion state



    def setState(self, D):
        """
        Arguments
        --------
        D: D weight
        """
        self.state = np.clip(D, self.state_min, self.state_max)



    def getOb(self):
        """
        Return
        ------
            A path.
        """
        ob = np.zeros((3,2))
        if self.state < 0.2: # too close
            ob[0] = [3, 2] # first point
            ob[1] = [5, 3] # second point
            ob[2] = [5, 3] # the last point
        elif self.state < 0.5: # medium
            ob[0] = [5, 2]
            ob[1] = [8, 3]
            ob[2] = [10, 10]
        else: # too far
            ob[0] = [7, 1]
            ob[1] = [9, 1]
            ob[2] = [10, 10]

        return ob



    def getLen(self, ob):
        """
        Arguments
        ---------
            ob: a path

        Return
        ------
            Path length
        """
        if not self.done(ob):
            return self.max_len

        last_pt = [0, 0]
        plen = 0.0
        for pt in ob:
            plen += np.linalg.norm(pt - last_pt)
            last_pt = pt

        return plen



    def getReward(self, ob):
        """
        Arguments
        ---------
            ob: a path

        Return
        ------
            Reward
        """
        r = self.max_len - self.getLen(ob)
        r = max(r, 0.0)
        return r



    def done(self, ob):
        """
        Arguments
        ---------
            ob: a path

        Return
        ------
            done or not
        """
        return (ob[len(ob)-1] == self.goal).all()



    def reset(self, init_state=None):
        """
        Set state randomly default and calculate corresponding path.

        Arguments
        ---------
            init_state: if it is None, then set state randomly

        Return
        ------
            A path.
        """
        if init_state is None:
            self.setState(np.random.rand())
        else:
            self.setState(init_state)

        return self.getOb()



    def decodeAction(self, action):
        """
        Arguments
        ---------
            action (int): encoded delta D weight.

        Return
        ------
            (float): Delta D weight.
        """
        return (action - 1) * 0.1



    def step(self, action):
        """
        Arguments
        ---------
            action: encoded delta D weight.

        Return
        ------
            *A path.
            *Reward: max_len - real_len.
            *Done: reach the goal or not.
            *Info.
        """
        dD = self.decodeAction(action)
        self.setState(self.state + dD)
        ob = self.getOb()

        return ob, self.getReward(ob), self.done(ob), {}