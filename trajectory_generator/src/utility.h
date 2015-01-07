#ifndef UTILITY_H
#define UTILITY_H
#include <iostream>
#include <vector>
#include <queue>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "ramp_msgs/TrajectoryRequest.h"
#include "ramp_msgs/Path.h"
#include <tf/transform_datatypes.h>
#include <ros/console.h>
#include "reflexxes_data.h"

#define PI 3.14159f

#define CYCLE_TIME_IN_SECONDS 0.1

enum TrajectoryType {
  ALL_STRAIGHT_SEGMENTS = 0,
  ALL_BEZIER            = 1,
  PARTIAL_BEZIER        = 2,
  TRANSITION            = 3,
  PREDICT               = 4
};


class Utility {
  public:
    Utility();
    
    const double positionDistance(const std::vector<double> a, const std::vector<double> b) const;
    const double positionDistance(const trajectory_msgs::JointTrajectoryPoint a, const trajectory_msgs::JointTrajectoryPoint b) const;

    const double findAngleFromAToB(const trajectory_msgs::JointTrajectoryPoint a, const trajectory_msgs::JointTrajectoryPoint b) const;
    const double findAngleFromAToB(const std::vector<double> a, const std::vector<double> b) const;
    const double findAngleFromAToB(const double x_prev, const double y_prev, const double x, const double y) const;
    const double findAngleToVector(const std::vector<double> p) const;
    
    const double findDistanceBetweenAngles(const double a1, const double a2) const;
    
    const double displaceAngle(const double a1, double a2) const;
    
    const double getEuclideanDist(const trajectory_msgs::JointTrajectoryPoint a, const trajectory_msgs::JointTrajectoryPoint b) const;
    const double getEuclideanDist(const std::vector<double> a, std::vector<double> b) const;

    const uint8_t getQuadrant(const double angle) const;
    const uint8_t getQuadrantOfVector(const std::vector<double> v) const;

    const ramp_msgs::Path getPath(const std::vector<ramp_msgs::MotionState> mps) const;
    const ramp_msgs::Path getPath(const std::vector<ramp_msgs::KnotPoint>   kps) const;

    const ramp_msgs::KnotPoint getKnotPoint(const ramp_msgs::MotionState ms) const;
    const trajectory_msgs::JointTrajectoryPoint getTrajectoryPoint(const ramp_msgs::MotionState ms) const;

    
    const std::string toString(const ramp_msgs::MotionState mp) const;
    const std::string toString(const ramp_msgs::KnotPoint kp) const;
    const std::string toString(const ramp_msgs::Path path) const;
    const std::string toString(const ramp_msgs::BezierInfo bi) const;
    const std::string toString(const ramp_msgs::RampTrajectory traj) const;
    const std::string toString(const trajectory_msgs::JointTrajectoryPoint p) const;
    const std::string toString(const ramp_msgs::TrajectoryRequest::Request tr) const;
};


#endif 
