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


    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;


      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&ModelStateTimeStampIntegration::OnUpdate, this));

      // m_nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, boost::bind(&ModelStateTimeStampIntegration::TimeStamp, this));
      model_state_subscriber.push_back(m_nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, boost::bind(&ModelStateTimeStampIntegration::TimeStamp, this, _1)));

      this->timestamp_integrated_publisher = m_nh.advertise< gazebo_modelstates_timestamp_patch::model_states_with_timestamp>("/gazebo_modelstates_with_timestamp", 10);
    }

    // Called by the world update start event
    public: void OnUpdate()
    {

    }
    public: void TimeStamp( const gazebo_msgs::ModelStates::ConstPtr  &current_model_states ){

      counter++;

      gz_time_now = this->model->GetWorld()->SimTime();



      gazebo_modelstates_timestamp_patch::model_states_with_timestamp my_test;

      my_test.gazebo_model_states = *current_model_states;
      my_test.gazebo_model_states_header.frame_id = "";
      my_test.gazebo_model_states_header.seq = counter;
      my_test.gazebo_model_states_header.stamp.sec = gz_time_now.sec;
      my_test.gazebo_model_states_header.stamp.nsec = gz_time_now.nsec;


      this->timestamp_integrated_publisher.publish(my_test);

      

    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelStateTimeStampIntegration)
}
