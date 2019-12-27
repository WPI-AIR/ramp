ramp
====

This is a ROS metapackage for Real-time Adaptive Motion Planning algorithm originally implemented by Sterling McLeod (https://github.com/sterlingm).

McLeod, Sterling, and Jing Xiao. "Real-time adaptive non-holonomic motion planning in unforeseen dynamic environments." 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016.

## To Run 
```
roscore
roslaunch ramp_launch gazebo_costmap.launch 
roslaunch ramp_launch planner_full_costmap_simulation.launch 
cd catkin_ws/src/RAMP_Gazebo/ramp_launch/
rosrun rviz rviz -d robot_costmap.rviz
```
In the planner terminal, the planner is waiting for the sensing module to get ready.
`rosparam set /ramp/sensing_ready true`

Waiting for Hilbert Map
`rosrun ramp_sensing load_occ_map.py`

Press `Enter` for the planner to begin first iteration.  
Press `Enter` for the planner to begin remainging iteration.

Publish random value on /combined_map topic.
```
rostopic pub /combined_map nav_msgs/OccupancyGrid "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
info:
  map_load_time: {secs: 0, nsecs: 0}
  resolution: 0.0
  width: 0
  height: 0
  origin:
    position: {x: 0.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}
data: [1]" 
```
