#!/bin/bash

# Fix Code
echo "== Gazebo patch: Replace Code =="
new_string="// do not refresh animation faster than 30 Hz sim time
  // if ((currentTime - this->prevFrameTime).Double() < (1.0 / 30.0))
  //  return;"

perl -i -p0e "s%// do not refresh.*?return;%$new_string%s" $HBP/gazebo/gazebo/physics/Actor.cc
echo "== Gazebo patch: Replace Code - DONE =="

# Build Gazebo
echo "== Gazebo patch: Build Gazebo =="
source $HBP/user-scripts/nrp_functions
build_gazebo
echo "== Gazebo patch: Build Gazebo - DONE  =="

echo "Gazebo Patch Done"
