ramp
====

This is a ROS metapackage for Real-time Adaptive Motion Planning algorithm originally implemented by Sterling McLeod (https://github.com/sterlingm).

McLeod, Sterling, and Jing Xiao. "Real-time adaptive non-holonomic motion planning in unforeseen dynamic environments." 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016.

## To Run 
```
roscore
roslaunch ramp_launch planner_parameters.launch
rosrun ramp_planner pub_map_odom
roslaunch gazebo_costmap.launch 
roslaunch ramp_launch planner_full_costmap_simulation.launch 
cd catkin_ws/src/RAMP_Gazebo/ramp_launch/
rosrun rviz rviz -d robot_costmap.rviz

