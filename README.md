# Robotics Programming Assignment


This repository hosts the source code for my robotics programming assignment, implemented in Python Programming and ROS2- Humble Framework. The assignment specifically addresses pothole detection using the LIMO robot, contributing to the evaluation of road inspection performance.
## Table of Contents

1. [Wiki](#wiki)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)


## Wiki

Explore the detailed documentation and guidelines in the [Wiki](https://github.com/arunabh-alt/Arunabh_Pothole_Assignment_27674565/wiki).

## Project Structure

The main code and packages are in the `Pothole-Assignment` folder. This folder contains `pothole`,`resource`,`test` folders and `package.xml`, `setup.cfg`,`setup.py` files. Here the main project codebase are available in the `pothole` folder. 

## ROS2-based Python Files

The core functionality of the assignment is implemented through Python-based ROS2 files. The following files contribute to different aspects of the assignment:

### 1. magenta_pothole

The `magenta_pothole` file is designed to detect simple potholes within a controlled environment. This file can count pothole, determine the sizes of potholes. This module focuses on identifying potholes in a specific setting.

### 2. real_object_detector

The `real_object_detector` file specializes in detecting and counting real-world potholes. It leverages Python programming and ROS2 to identify and analyze potholes in a broader context.

### 3. analysis
The `analysis` code analyze the pothole based on the sizes. Determine the pothole whether it is bad condition or big pothole , middle size pothole ,minor condition pothole.
## Getting Started

To explore the codebase and run the assignment, follow these steps:


1. **Install ROS2 on your Ubuntu 22.04 System**
2. **Create a folder (e.g- `assignment_limo`) and then make a directory name `src`** 
3. **Clone the Repository to the `src` folder:**
   ```bash
   git clone https://github.com/arunabh-alt/Arunabh_Pothole_Assignment_27674565.git
  
### Build the package:
    colcon build --symlink-install
### Create a new terminal , and run the command: 
    cd assignment_limo
    source install\setup.bash  
### Launch the node
    ros2 run package_name node_name

## Simulation 
### Launch Gazebo
    
    git clone https://github.com/LCAS/CMP9767_LIMO.git
    ros2 launch limo_gazebosim limo_gazebo_diff.launch.py world:=src/CMP9767_LIMO/assignment_template/worlds/potholes_simple.world
### Launch Rviz
    
    ros2 launch limo_navigation limo_navigation.launch.py use_sim_time:=true map:=src/CMP9767_LIMO/assignment_template/maps/potholes_20mm.yaml params_file:=src/CMP9767_LIMO/assignment_template/params/nav2_params.yaml
