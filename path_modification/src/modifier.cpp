#include "modifier.h"

Modifier::Modifier(ramp_msgs::ModificationRequest::Request& req) : mod_req(req) {}


Modifier::Modifier(ramp_msgs::ModificationRequest::Request& req, std::vector<ramp_msgs::Range> r) : mod_req(req), ranges_(r) 
{
  in_.ranges_ = r;
  chg_.ranges_ = r;
  repair_.ranges_ = r;

  /*ROS_INFO("Ranges in Path Mod package Modifier");
  for(int i=0;i<ranges_.size();i++)
  {
    ROS_INFO("Range %i min: %f max %f", i, ranges_[i].min, ranges_[i].max);
  }*/
}

const std::vector<ramp_msgs::Path> Modifier::perform() 
{
  std::vector<ramp_msgs::Path> result;

  /*ROS_INFO("Modifier received path: %s", u.toString(mod_req.paths[0]).c_str());
  ROS_INFO("mod_req.op: %s", mod_req.op.c_str());
  
  for(int i=0;i<mod_req.paths.size();i++)
  {
    for(int j=0;j<mod_req.paths[i].points.size();j++)
    {
      if(mod_req.paths[i].points[j].motionState.positions[1] > 2.0)
      {
        ROS_INFO("Path has an out-of-bounds motion state");
      }
    }
  }*/

  if(mod_req.op == "insert") 
  {
    in_.path_ = mod_req.paths.at(0); 
    result.push_back(in_.perform());
  }

  else if(mod_req.op == "delete") 
  {
    del_.path_ = mod_req.paths.at(0);
    result.push_back(del_.perform());
  }
  
  else if(mod_req.op == "change") 
  {
    chg_.path_ = mod_req.paths.at(0);
    result.push_back(chg_.perform());
  }

  else if(mod_req.op == "swap") 
  {
    swap_.path_ = mod_req.paths.at(0);
    result.push_back(swap_.perform());
  }

  else if(mod_req.op == "crossover") 
  {
    cross_.path1_ = mod_req.paths.at(0);
    cross_.path2_ = mod_req.paths.at(1);
    result = cross_.perform();
  }

  else if(mod_req.op == "repair")
  {
    repair_.path_ = mod_req.paths[0];
    repair_.dir_  = mod_req.repair_dir;
    repair_.dist_ = mod_req.repair_dist;
    repair_.r_    = mod_req.repair_ob_r;

    result.push_back(repair_.perform());
  }

  //ROS_INFO("Modifier returning: %s", u.toString(result.at(0)).c_str());
  
  for(int i=0;i<result.size();i++)
  {
    for(int j=0;j<result[i].points.size();j++)
    {
      if(result[i].points[j].motionState.positions[0] > ranges_[0].max ||
          result[i].points[j].motionState.positions[1] > ranges_[1].max ||
          result[i].points[j].motionState.positions[0] < ranges_[0].min ||
          result[i].points[j].motionState.positions[1] < ranges_[0].min)
      {
        ROS_INFO("Result Path has an out-of-bounds motion state");
      }
    }
  }


  return result;
}
