# NRP_grasping
ROS packages and models for learning to grasp in NRP using shadow hand and UR10

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
git submodule init
git submodule update --recursive
cp GazeboRosPackages/* $HBP/GazeboRosPackages/src -a
cp Models/* $HBP/Models/ -a # add folder names in models.txt if needed
cp Experiments/* $HBP/Experiments -a
cd $HBP/Models
./create-symlinks.sh
```

```
cd $HBP/GazeboRosPackages
catkin_make
```

Adjust `GAZEBO_MODEL_PATH` and `MESH_WORKSPACE_PATH` in `smart_grasping_sandbox/launch/smart_grasping_sandbox.launch`


## Run
1. Starte the Neurorobotics Platform
  ```
  cle-start
  cle-frontend
  ```
2. Load robot description, start moveit and gazebo2moveit nodes
  ```
  roslaunch smart_grasping_sandbox smart_grasping_sandbox.launch
  ```
3. Open the shadow experiment (do not start the simulation yet)
4. Load controllers
  ```
  roslaunch fh_desc controllers.launch
  ```
5. Start simulation in NRP
6. Fire up python sample RL python script
  ```
  roscd smart_grasping_sandbox/scripts
  python RL-hand.py
  ```
