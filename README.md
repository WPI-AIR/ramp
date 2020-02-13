Ramp
====

This is a ROS metapackage for Real-time Adaptive Motion Planning algorithm originally implemented by Sterling McLeod (https://github.com/sterlingm).

McLeod, Sterling, and Jing Xiao. "Real-time adaptive non-holonomic motion planning in unforeseen dynamic environments." 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016.

![](/results/ramp_gazebo.gif)

## Deep Ramp
Deep-RAMP uses Deep Learning to optimize the cost function that finds the best trajectory in the population.   

## To Run 
```
roslaunch ramp_launch gazebo_costmap.launch 
roslaunch ramp_launch planner_full_costmap_simulation.launch 
cd catkin_ws/src/RAMP_Gazebo/ramp_launch/
rosrun rviz rviz -d robot_costmap.rviz
```
The robot destination can be changed in /ramp_launch/config/robot_0.yaml
