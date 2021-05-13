#include "evaluate.h"

Evaluate::Evaluate() : orientation_infeasible_(0), T_norm_(50.0), A_norm_(PI), _1_D_norm_(1.0), coll_time_norm_(zero),
                       last_T_weight_(-1.0), last_A_weight_(-1.0), last_D_weight_(-1.0), last_Q_coll_(-1.0), last_Q_kine_(-1.0), 
                       Ap(1.0), Bp(1.0), L(1.0) {}

void Evaluate::set_ped_pose(geometry_msgs::Pose& pose_){
  ped_pose = pose_;
  // std::cout<< pose_.position.x << std::endl;
}

void Evaluate::set_robot_pose(geometry_msgs::Pose& pose_){
  robot_pose = pose_;
  // std::cout<< pose_.position.x << std::endl;
}

float Evaluate::get_dp(float x, float y){
  // return sqrt(pow((robot_pose.position.x - ped_pose.position.x), 2) +
  //                      pow((robot_pose.position.y - ped_pose.position.y), 2));
  return sqrt(pow((robot_pose.position.x - x), 2) +
                     pow((robot_pose.position.y - y), 2));
  // std::cout<< dp << std::endl;
}

geometry_msgs::Point Evaluate::get_np(float x, float y){
  float dp = get_dp(x, y);

  geometry_msgs::Point np; 
  np.x = (robot_pose.position.x - ped_pose.position.x)/dp;
  np.y = (robot_pose.position.y - ped_pose.position.y)/dp;
  
  return np;
  // return sqrt(pow((np_.x), 2) +
  //                      pow((np_.y), 2));
  // std::cout<< np_.x << std::endl;
}

float Evaluate::getMinObsDistance(float x, float y, trajectory_msgs::JointTrajectory& Traj){
  float d_min = 100;
  float dist = 100;
  for(int i=0; i<Traj.points.size(); i++){
    float tx = Traj.points[i].positions[0];
    float ty = Traj.points[i].positions[1];

    dist = sqrt(pow((tx - x), 2) +
                     pow((ty - y), 2));
    
    if(dist < d_min)
    {
      d_min = dist;
    }
  }

  return d_min;
}

void Evaluate::perform(ramp_msgs::EvaluationRequest& req, ramp_msgs::EvaluationResponse& res)
{
  //ROS_INFO("In Evaluate::perform()");
  //ros::Time t_start = ros::Time::now();
  
  /*
   * Initialize some class memebers
   */
  // Set orientation members
  orientation_.currentTheta_  = req.currentTheta;
  orientation_.theta_at_cc_   = req.theta_cc;

  // Set imminent collision
  imminent_collision_ = req.imminent_collision;
  //ROS_INFO("imminent_collision_: %s", imminent_collision_ ? "True" : "False");

  // Reset orientation_infeasible for new trajectory
  orientation_infeasible_ = false;

  /*
   * Compute feasibility
   */
  performFeasibility(req);

  // Set response members
  // req.trajectory.feasible = !qr_.collision_ && !orientation_infeasible_;
  // res.feasible = !qr_.collision_ && !orientation_infeasible_;
  req.trajectory.feasible = !qr_.collision_;
  res.feasible = !qr_.collision_;
  req.trajectory.feasible = res.feasible;
  //ROS_INFO("qr_.collision: %s orientation_infeasible_: %s", qr_.collision_ ? "True" : "False", orientation_infeasible_ ? "True" : "False");
  //////ROS_INFO("performFeasibility: %f", (ros::Time::now()-t_start).toSec());

  if(qr_.collision_)
  {
    ////////ROS_INFO("Not feasible");
    res.t_firstCollision = ros::Duration(qr_.t_firstCollision_);
  }
  else
  {
    ////////ROS_INFO("Feasible");
    res.t_firstCollision = ros::Duration(9999.f);
  }


  /*
   * Compute fitness
   */
  if(req.full_eval)
  {
    ////ROS_INFO("Requesting fitness!");
    performFitness(req.trajectory, req.offset, res.fitness, res.min_obs_dis, res);
  }
  //////ROS_INFO("performFitness: %f", (ros::Time::now()-t_start).toSec());
}


// Redo this method at some point
// It's modiftying trj AND returning a value
void Evaluate::performFeasibility(ramp_msgs::EvaluationRequest& er) 
{
  ////ROS_INFO("In Evaluate::performFeasibility");
  ros::Time t_start = ros::Time::now();

  // Check collision
  ros::Time t_numeric_start = ros::Time::now();
  cd_.performNum(er.trajectory, er.obstacle_trjs, er.robot_radius, er.obstacle_radii, qr_);
  ros::Duration d_numeric   = ros::Time::now() - t_numeric_start;
  t_numeric_.push_back(d_numeric);

  ////ROS_INFO("result.collision: %s", qr_.collision_ ? "True" : "False");
  /*ros::Time t_analy_start = ros::Time::now();
  cd_.perform(er.trajectory, er.obstacle_trjs, qr_);
  ros::Duration d_analy = ros::Time::now() - t_analy_start;
  t_analy_.push_back(d_analy);*/

  ////////ROS_INFO("feasible: %s", er.trajectory.feasible ? "True" : "False");
  er.trajectory.feasible            = !qr_.collision_;
  er.trajectory.t_firstCollision    = ros::Duration(qr_.t_firstCollision_);

  ramp_msgs::RampTrajectory* trj = &er.trajectory;
  
  if((!er.imminent_collision && er.consider_trans && !er.trans_possible) ||
      orientation_.getDeltaTheta(*trj) > 1.5708) // 90 degree
  {
    ////ROS_INFO("In final if statement");
    orientation_infeasible_ = true;
  }
  else
  {
    ////ROS_INFO("er.imminent_collision: %s er.consider_trans: %s er.trans_possible: %s", er.imminent_collision ? "True" : "False", er.consider_trans ? "True" : "False", er.trans_possible ? "True" : "False");
  }
  
  ////////////ROS_INFO("performFeasibility time: %f", (ros::Time::now() - t_start).toSec());
}



/** This method computes the fitness of the trajectory_ member */
void Evaluate::performFitness(ramp_msgs::RampTrajectory& trj, const double& offset, double& result,
                            double& min_obs_dis, ramp_msgs::EvaluationResponse& res) 
{
  //////ROS_INFO("In Evaluate::performFitness");
  ros::Time t_start = ros::Time::now();
  
  if(trj.feasible)
  {
    //ROS_INFO("In if(feasible)");
    
    // Get total time to execute trajectory
    double T = trj.trajectory.points.at(trj.trajectory.points.size()-1).time_from_start.toSec();

    /*
     * Trajectory point generation ends at the end of the non-holonomic segment
     * For the remaining segments, estimate the time and orientation change needed to execute them
     */

    // p = last non-holonomic point on trajectory
    trajectory_msgs::JointTrajectoryPoint p = trj.trajectory.points.at(trj.trajectory.points.size()-1);
    //////////ROS_INFO("p: %s", utility_.toString(p).c_str());

    // Find knot point index on holonomic path where non-holonomic segment ends
    uint16_t i_end=0;
    for(uint16_t i=0;i<trj.holonomic_path.points.size();i++)
    {
      //////////ROS_INFO("i: %i trj.holonomic_path.points.size(): %i", (int)i, (int)trj.holonomic_path.points.size());
      double dist = utility_.positionDistance(trj.holonomic_path.points[i].motionState.positions, p.positions);

      ////ROS_INFO("trj.holonomic_path[%i]: %s", (int)i, utility_.toString(trj.holonomic_path.points[i].motionState).c_str());
      ////ROS_INFO("dist: %f", dist);
      ////ROS_INFO("offset: %f", offset);

      // Account for some offset
      if( dist*dist < (0.2 + offset) )
      {
        i_end = i; 
        break;
      }
    } // end for
    
    //////ROS_INFO("i_end: %i", (int)i_end);
    //////ROS_INFO("trj.holonomic_path.points.size(): %i", (int)trj.holonomic_path.points.size());

    // For each segment in remaining holonomic path,
    // accumulate the distance and orientation change needed for remaining segment
    double dist=0;
    double delta_theta=0;
    double last_theta = p.positions[2];
    for(uint8_t i=i_end;i<trj.holonomic_path.points.size()-1;i++)
    {
      //////////ROS_INFO("i: %i", (int)i);
      dist += utility_.positionDistance(trj.holonomic_path.points[i].motionState.positions, trj.holonomic_path.points[i+1].motionState.positions);
      
      double theta = utility_.findAngleFromAToB(trj.holonomic_path.points[i].motionState.positions, trj.holonomic_path.points[i+1].motionState.positions);
      
      delta_theta += fabs(utility_.findDistanceBetweenAngles(last_theta, theta));
      
      last_theta = theta;
    }
    //////ROS_INFO("dist: %f delta_theta: %f", dist, delta_theta);

    double max_v = 0.25 / 2.0;
    double max_w = PI / 8.0;

    // Estimate how long to execute positional and angular displacements based on max velocity
    double estimated_linear   = dist / max_v;
    double estimated_rotation = delta_theta / max_w;

    //////ROS_INFO("estimated_linear: %f estimated_rotation: %f", estimated_linear, estimated_rotation);

    T += (estimated_linear + estimated_rotation);
    res.time_fitness = T;
    if (T < zero) T = zero;
    double _1_T = 1.0 / T;

    // Orientation
    double A = fabs(orientation_.perform(trj));
    res.orien_fitness = A;
    if (A < zero) A = zero;
    double _1_A = 1.0 / A;

    /*double v = sqrt( pow(trj.trajectory.points[0].velocities[0],2) + pow(trj.trajectory.points[1].velocities[1],2) );
    A_weight_ = v;*/

    // Minimum distance to any obstacle
    double D = cd_.min_dist_;
    res.obs_fitness = D;
    if (D < zero) D = zero;
    double _1_D = 1.0 / D; // consider 1/D, not D
    min_obs_dis = D;
    
    //ROS_INFO("T: %f A: %f D: %f", T, A, D);

    // Update normalization for Time if necessary
    // if(T > T_norm_)
    // {
    //   T_norm_ = T;
    // }
    // if(A > A_norm_)
    // {
    //   A_norm_ = A;
    // }
    // if(_1_D > _1_D_norm_)
    // {
    //   _1_D_norm_ = _1_D;
    // }

    // Normalize terms
    T /= T_norm_;
    A /= A_norm_;
    _1_D /= _1_D_norm_;
    
    
    Ap /= Ap_norm_;
    Bp /= Bp_norm_;
    L /= L_norm_;
    
    //ROS_INFO("Normalized terms T: %f A: %f D: %f", T, A, D);

    // Weight terms
    // T_weight_ = 1.0;
    if (!ros::param::get("/ramp/eval_weight_T", T_weight_)) {
      // if fail to get the parameter
      T_weight_ = 1.0; // set it to the default
    }
    static bool is_set_T = false;
    if (!is_set_T) {
      ros::param::set("/ramp/eval_weight_T", T_weight_);
      is_set_T = true;
    }

    if (!ros::param::get("/ramp/eval_weight_A", A_weight_)) {
      // if fail to get the parameter
      A_weight_ = 0.005; // set it to the default
    }
    static bool is_set_A = false;
    if (!is_set_A) {
      ros::param::set("/ramp/eval_weight_A", A_weight_);
      is_set_A = true;
    }

    if (!ros::param::get("/ramp/eval_weight_D", D_weight_)) {
      // if fail to get the parameter
      D_weight_ = 0.4; // set it to the default
    }
    static bool is_set_D = false;
    if (!is_set_D) {
      ros::param::set("/ramp/eval_weight_D", D_weight_);
      is_set_D = true;
    }

        Ap_weight_ = 1.0;
    if (!ros::param::get("/ramp/eval_weight_Ap", Ap_weight_)) {
      // if fail to get the parameter
      Ap_weight_ = 1.0; // set it to the default
    }
    static bool is_set_Ap = false;
    if (!is_set_Ap) {
      ros::param::set("/ramp/eval_weight_Ap", Ap_weight_);
      is_set_Ap = true;
    }

    Bp_weight_ = 1.0;
    if (!ros::param::get("/ramp/eval_weight_Bp", Bp_weight_)) {
      // if fail to get the parameter
      Bp_weight_ = 1.0; // set it to the default
    }
    static bool is_set_Bp = false;
    if (!is_set_Bp) {
      ros::param::set("/ramp/eval_weight_Bp", Bp_weight_);
      is_set_Bp = true;
    }

    L_weight_ = 1.0;
    if (!ros::param::get("/ramp/eval_weight_L", L_weight_)) {
      // if fail to get the parameter
      L_weight_ = 1.0; // set it to the default
    }
    static bool is_set_L = false;
    if (!is_set_L) {
      ros::param::set("/ramp/eval_weight_L", L_weight_);
      is_set_L = true;
    }
    // T *= T_weight_;
    // A *= A_weight_;
    // D *= D_weight_; // this is wrong! this is 1.0 / (D * D_weight_), but what we need is D * (1.0 / D_weight_)
    
    //ROS_INFO("Weighted terms T: %f A: %f D: %f", T, A, D);
    // Compute Interaction Force
    double F = zero;
    double fgoal = zero;
    // Compute overall cost
    double cost = zero;

    // Get Euclidean distance of the ped from the robot
    float rp = 1.0;
    result = 100000; 

    // float obs1x = 0.5;
    // float obs1y = 4.5;
    float obs1x = ped_pose.position.x;
    float obs1y = ped_pose.position.y;
    float obs2x = 2.5;
    float obs2y = 4.5;

    std::vector<double> goal {5, 5, 0.784}; // initialization
    
    ros::param::get("robot_info/goal", goal);
    
    float ex = goal[0] - robot_pose.position.x;
    float ey = goal[1] - robot_pose.position.y;

    geometry_msgs::Point np1 = get_np(obs1x, obs1y);
    float dp1 = getMinObsDistance(obs1x, obs1y, trj.trajectory);
    float cos_phi_1 = - (ex*np1.x + ey*np1.y)/sqrt(ex*ex + ey*ey); 
    double omega_1 = L_weight_ + (1-L_weight_)*(1 + cos_phi_1)/2; // = 1 - (0.5to1)L
    float Fx_1 = Ap_weight_*exp((rp- dp1)/(Bp_weight_)) * omega_1 *(-np1.x);
    float Fy_1 = Ap_weight_*exp((rp- dp1)/(Bp_weight_)) * omega_1 *(-np1.y);

    geometry_msgs::Point np2 = get_np(obs2x, obs2y);
    float dp2 = getMinObsDistance(obs2x, obs2y, trj.trajectory);
    float cos_phi_2 = - (ex*np2.x + ey*np2.y)/sqrt(ex*ex + ey*ey); 
    double omega_2 = L_weight_ + (1-L_weight_)*(1 + cos_phi_2)/2; // = 1 - (0.5to1)L
    float Fx_2 = Ap_weight_*exp((rp- dp2)/(Bp_weight_)) * omega_2 *(-np2.x);
    float Fy_2 = Ap_weight_*exp((rp- dp2)/(Bp_weight_)) * omega_2 *(-np2.y);

    // std::cout << "omega1: " << omega_1 << " np1.x: " << np1.x << "np1.y" << np2.x << std::endl;
    // std::cout << "dp1: " << dp1 << " dp2: " << dp2 << " min_obs_dis: " << min_obs_dis << std::endl;

    float F_max = 10;
    float F1 = F_max*sqrt(pow((Fx_1), 2) + pow((Fy_1), 2));
    float F2 = 0*F_max*sqrt(pow((Fx_2), 2) + pow((Fy_2), 2));
    
    F += F1;
    F += F2;

    if (F>F_max)
      F = F_max;

    ros::param::set("/ramp/Fint", F);
    ros::param::set("/ramp/dp_obs", dp);

    // std::cout << "dp : " << dp << std::endl;  
    // F += Ap_weight_*exp((rp- (dp_weight_* dp))/(Bp_weight_* Bp)) * omega *(-np);
    // std::cout << "T_cost: " << T_weight_ * T  << " F: " << F << " F1: " << F1 << " F2: " << F2 << std::endl;  
    cost += T_weight_ * T + A_weight_ * A + 0*D_weight_ * _1_D + F;
    result = 1.0 / cost;

    // result = 100000; // must be larger than infeasible
    // result += T_weight_ * _1_T + A_weight_ * _1_A + D_weight_ * D;
  }

  else // infeasible
  {
    //ROS_INFO("In else(infeasible)"); 
    
    // penalties += orientation_.getPenalty();
    
    ////////////ROS_INFO("trj.t_firstColl: %f", trj.t_firstCollision.toSec());

    // Q_coll_ = 1.0;
    if (!ros::param::get("/ramp/eval_weight_Qc", Q_coll_)) {
      Q_coll_ = 1.0; // default
    }
    static bool is_set_Qc = false;
    if (!is_set_Qc) {
      ros::param::set("/ramp/eval_weight_Qc", Q_coll_);
      is_set_Qc = true;
    }

    double coll_time = trj.t_firstCollision.toSec(); // larger, better
    if (coll_time < zero) {
      coll_time = zero;
    }
    double _1_coll_time = 1.0 / coll_time;
    if (_1_coll_time > _1_coll_time_norm_) {
      _1_coll_time_norm_ = _1_coll_time;
    }
    _1_coll_time /= _1_coll_time_norm_;

    // If infeasible due to orientation change (i.e. no smooth curve)
    // If there is imminent collision, do not add this penalty (it's okay to stop and rotate)
    double delta_theta;
    double _1_delta_theta;
    if(1 || orientation_infeasible_)// && !imminent_collision_)
    {
      delta_theta = fabs(orientation_.getDeltaTheta(trj) < 0.11 ? 0.1 : orientation_.getDeltaTheta(trj));
      if (delta_theta < zero) {
        delta_theta = zero;
      }
      _1_delta_theta = 1.0 / delta_theta;

      if (!ros::param::get("/ramp/eval_weight_Qk", Q_kine_)) {
        Q_kine_ = 1.0; // set it to the default
      }
      static bool is_set_Qk = false;
      if (!is_set_Qk) {
        ros::param::set("/ramp/eval_weight_Qk", Q_kine_);
        is_set_Qk = true;
      }

      // if (delta_theta > A_norm_) {
      //   A_norm_ = delta_theta;
      // }
      delta_theta /= A_norm_;
    }

    double penalties = 1.0; // for infeasible trajectory, penalties cannot be zero
    penalties += Q_coll_ * _1_coll_time + Q_kine_ * delta_theta;
    penalties *= 10000.0; // the fitness of infeasible trajextory must be smaller than feasible trajectory
    result = 1.0 / penalties;
    // result = Q_coll_* coll_time + Q_kine_ * _1_delta_theta;
  }

  // if weights change, print them
  if (Ap_weight_ != last_Ap_weight_ ||
      Bp_weight_ != last_Bp_weight_ ||
      L_weight_ != last_L_weight_ ||
      Q_coll_   != last_Q_coll_   ||
      Q_kine_   != last_Q_kine_) {
        printf("weights have changed to: Ap = %.3lf, Bp = %.5lf, L = %.3lf, Qc = %.3lf, Qk = %.3lf\n",
          Ap_weight_, Bp_weight_, L_weight_, Q_coll_, Q_kine_);
        last_Ap_weight_ = Ap_weight_; last_Bp_weight_ = Bp_weight_; 
        last_L_weight_ = L_weight_; last_Q_coll_   = Q_coll_;   last_Q_kine_   = Q_kine_;
      }

  // ROS_INFO("performFitness time: %f", (ros::Time::now() - t_start).toSec());
  // ROS_INFO("Exiting Evaluate::performFitness");
} // End performFitness
