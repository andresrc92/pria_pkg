# ROS 2 Package to generate new datasets, train PyTorch model on it and use it in a robotic manipulation task.

This repository contains the code for my **Robotics and AI Specialization** capstone project. It also contains code prototypes and tests, written by me as part of my learning and experimentation.

The package works both in simulation or with a real robot, supported manipulator is Universal Robots UR5e.  

## For use with simulated robot:

Run NVIDIA Isaac Sim:  

    ~/.local/share/ov/pkg/isaac-sim-4.0.0/isaac-sim.sh

Open USD located in:  
    
    /usd/robot2bin.usd

Run Docker image of URSim:  

    docker run --rm -it -p 5900:5900 -p 6080:6080 -v ${HOME}/.ursim/urcaps:/urcaps -v ${HOME}/.ursim/programs:/ursim/programs --name ursim universalrobots/ursim_e-series


Launch ROS 2 ur_robot_driver package:  

    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=172.17.0.2 launch_rviz:=true


## For use with real robot:


Connect the robot controller to the PC through ethernet  
Set controller IP accordingly  
Launch ROS 2 ur_robot_driver package:  

    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.100.2 launch_rviz:=true

Launch a camera node  

    ros2 run pria usb_camera.py


## Once you're setup in either stage (sim or real) you can start creating a dataset:

Place an object somewhere withing the robot's reach  
Jog the robot to the position relative to the object where the grasping should occur  
Start the data collection node with the object's name as parameter  

    ros2 run pria data_collection --ros-args -p folder:=<object name>

This will start to move the robot around the object, saving the images and the robot position relative to the goal position. In total it will create 4 data folders, each with the images taken during 4 different trajectories around the object.


## Model training

The model training script isn't actually a ROS node, but it's located in this package.  

    python3 pytorch_training_script.py '<object name>' 100

This script will train 4 models on the 4 datasets created for the object and store them in the same folders where the data is.  


## Model usage

Once all models are trained, running a grasp task is done like so:  

    ros2 run pria grasping_task_sim --ros-args -p folder:=<object name>

This node will use 2 PyTorch models, in a 3 stage task. First it will use a model for approaching the object from the distance until the model prediction is small enough to consider the robot is already close to where it should be.

Then, a refinement stage uses a model trained on data taken close to the object to correct any XY and YAW differences before moving on to the final stage where it will attempt a grasp.
