# Robotics Programming Assignment

This repository contains the source code for my robotics programming assignment, implemented in Python. The assignment focuses on the LIMO robot for performance evaluation.

## Project Structure

The main codebase is organized in the `src/path/to/submodule` directory. For detailed specifications, refer to this path.

## ROS2-based Python Files

The core functionality of the assignment is implemented through Python-based ROS2 files. The following files contribute to different aspects of the assignment:

### 1. magenta_pothole

The `magenta_pothole` file is designed to detect simple potholes within a controlled environment. This module focuses on identifying potholes in a specific setting.

### 2. object_detector

The `object_detector` file specializes in detecting real-world potholes. It leverages Python programming and ROS2 to identify and analyze potholes in a broader context.

## Getting Started

To explore the codebase and run the assignment, follow these steps:


1. **Install ROS2 on your Ubuntu 22.04 System**

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/arunabh-alt/Arunabh_Pothole_Assignment_27674565.git
3. **Make sure clone the repo on your folder's src directory**   
### Build the package:
    colcon build --symlink-Install
### Create a new terminal , and run the command:  
    source install\setup.bash  
### Run the node
    cd <folder>
    ros2 run package_name node_name
