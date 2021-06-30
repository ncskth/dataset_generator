./gazebo_actor-updaterate_patch.sh
cd /opt/dataset_generator
cp -r Models/* $HBP/Models
cp -r GazeboRosPkgs/src/* $HBP/GazeboRosPackages/src/
cd $HBP/Models
./create-symlinks.sh
cd $HBP/GazeboRosPackages
rm -rf build devel
catkin build

pip3 install texttable future