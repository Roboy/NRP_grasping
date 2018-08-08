# NRP_grasping
ROS packages and models for learning to grasp in NRP using shadow hand and UR5

## Prerequisites
```
sudo apt install ros-kinetic-controller-manager ros-kinetic-ros-controllers ros-kinetic-moveit
```

```
pip2.7 install tensorflow keras h5py sklearn bokeh bayesian-optimization pandas
```
## Setup & compile
```
cd NRP_grasping
cp GazeboRosPackages/* $HBP/GazeboRosPackages/src -a
cp Models/* $HBP/Models/ -a # add folder names in models.txt if needed
cp Experiments/* $HBP/Experiments -a
```

```
cd $HBP/GazeboRosPackages
catkin_make
```

## Run
1. Open the shadow experiment (do not start the simulation yet)
2. Load robot description, start moveit and gazebo2moveit nodes
  ```
  roslaunch smart_grasping_sandbox smart_grasping_sandbox.launch
  ```
3. Load controllers
  ```
  roslaunch fh_desc controllers.launch
  ```
4. Start simulation in NRP
5. Fire up python sample RL python script
  ```
  rosrun smart_grasping_sandbox RL-grasp.py
  ```
