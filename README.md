# Installation instructions

## Clone experiment to your preferred location
(do not clone into your nrpStorage directly)

```
cd /your_experiment_path
git clone https://github.com/ncskth/dataset_generator.git
cd dataset_generator
```

## symlink experiment into nrpStorage
```
ln -s /your_experiment_path/dataset_generator/NRPExp_DVSDatabaseGenerator $HOME/.opt/nrpStorage/NRPExp_DVSDatabaseGenerator
```

## Copy Model files
```
cp -r Models/* $HBP/Models
```

## Copy ROS packages (contains Kuka IIWA MoveIt configuration)
```
cp -r GazeboRosPkgs/src/* $HBP/GazeboRosPackages/src/
```

## Link new models
```
cd $HBP/Models
./create-symlinks.sh
```

## Compile ROS Packages
```
cd $HBP/GazeboRosPkgs
catkin_make -j8 
```

## Fix Gazebo
```
The update frequency of Gazebo actors is only 30Hz, on order to remove the limit of the actor update frequency run the 
./gazebo-actor-updaterate_patch.sh
```

## rosbag record
Default rosbag record path is $HOME/database_generator_record_TIME.bag
you can change the path in NRPExp_DVSDatabaseGenerator/rosbag_record.launch

# Rosbag useful links
http://wiki.ros.org/rosbag
