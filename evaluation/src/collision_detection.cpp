#include "collision_detection.h"

/** Returns true if trajectory_ is in collision with any of the objects */
const bool CollisionDetection::perform() const {
  std::cout<<"\nobstacle_list.size(): "<<obstacle_list.size();
  
  //Go through each point in the trajectory 
  //and check if it is in collision with any of the objects
  for(unsigned int i=0;i<trajectory_.trajectory.points.size();i++) {
  
    trajectory_msgs::JointTrajectoryPoint p = trajectory_.trajectory.points.at(i);
    double p_x = p.positions.at(0);
    double p_y = p.positions.at(1);

    //Check the point against each obstacle
    for(unsigned int o=0;o<obstacle_list.size();o++) {
      if( (p_x >= obstacle_list.at(o).x1) && (p_x <= obstacle_list.at(o).x2) 
        && (p_y >= obstacle_list.at(o).y1) && (p_y <= obstacle_list.at(o).y2) ) 
        {
          return true;
        }      
    }
  }
  
  return false;  
}
