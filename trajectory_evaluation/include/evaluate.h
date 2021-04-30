#ifndef EVALUATE_H
#define EVALUATE_H
#include "ramp_msgs/EvaluationSrv.h"
#include "euclidean_distance.h"
#include "orientation.h"
#include "collision_detection.h"
#include "utility.h"
#include "ramp_msgs/PedSim.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"




class Evaluate {
  public:
    Evaluate();

    void perform(ramp_msgs::EvaluationRequest& req, ramp_msgs::EvaluationResponse& res);
    void performFeasibility(ramp_msgs::EvaluationRequest& er);
    void performFitness(ramp_msgs::RampTrajectory& trj, const double& offset, double& result, double& min_obs_dis, ramp_msgs::EvaluationResponse& res);
    void pedsimParams(const ramp_msgs::PedSim& msg);
    float get_dp();
    void get_np();

    geometry_msgs::Pose robot_pose;
    geometry_msgs::Pose ped_pose;
    geometry_msgs::Point np_;  // To do rename to np_ after removing the other np
    /** Different evaluation criteria */
    EuclideanDistance eucDist_;
    Orientation orientation_;

    ramp_msgs::EvaluationResponse res_;
    
    CollisionDetection cd_;
    CollisionDetection::QueryResult qr_;

    //Information sent by the request
    ramp_msgs::RampTrajectory trajectory_;
    std::vector<ramp_msgs::RampTrajectory> ob_trjs_;


    double last_Q_coll_;
    double last_Q_kine_;
    double Q_coll_;
    double Q_kine_;

    bool imminent_collision_;

    double T_norm_;
    double _1_T_norm_;
    double A_norm_;
    double _1_A_norm_;
    double D_norm_;
    double _1_D_norm_;
    double coll_time_norm_;
    double _1_coll_time_norm_;

    double Ap_norm_;
    double Bp_norm_;
    double dp_norm_;
    double L_norm_;
    double k_norm_;

    double Ap;
    double Bp;
    double dp;
    double L;
    double k;
    double rp;
    double np;

    double last_T_weight_;
    double last_A_weight_;
    double last_D_weight_;

    double last_Ap_weight_;
    double last_Bp_weight_;
    double last_dp_weight_;
    double last_L_weight_;
    double last_k_weight_;

    double T_weight_;
    double A_weight_;
    double D_weight_;

    double Ap_weight_;
    double Bp_weight_;
    double dp_weight_;
    double L_weight_;
    double k_weight_;


    double max_speed_linear;
    double max_speed_angular;

    std::vector< ros::Duration > t_analy_;
    std::vector< ros::Duration > t_numeric_;
  private:
    Utility utility_;
    bool orientation_infeasible_;
    const double zero = 0.00001;
};

#endif
