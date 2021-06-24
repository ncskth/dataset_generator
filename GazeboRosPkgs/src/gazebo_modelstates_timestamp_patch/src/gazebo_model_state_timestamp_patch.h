#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo_msgs/ModelStates.h>
#include <boost/bind.hpp>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <gazebo_modelstates_timestamp_patch/model_states_with_timestamp.h>


namespace gazebo
{
  class ModelStateTimeStampIntegration : public ModelPlugin
  {

       // ROS node handle
    public: ros::NodeHandle m_nh;
      // ROS subscriber for joint control values
    std::vector<ros::Subscriber> model_state_subscriber;
    ros::Publisher timestamp_integrated_publisher;
    gazebo::common::Time gz_time_now;
    unsigned long int counter = 0;


    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/);

    // Called by the world update start event
    public: void OnUpdate();
    public: void TimeStamp( const gazebo_msgs::ModelStates::ConstPtr  &current_model_states );

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };
  
}
