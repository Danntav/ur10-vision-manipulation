# robotic-arm-5DoF: A Simulation-First Approach

A 5DoF robotic arm built from scratch as part of my Computer Engineering final project (TCC). This repository documents the entire development process, with a strong emphasis on a simulation-first methodology, encompassing design, electronics, mechanics, firmware, and testing.

## Objectives

- Design and 3D print the mechanical structure of the robotic arm.
- **Develop and validate the entire system in a simulated environment (Gazebo with ROS2) as the primary development phase.**
- Implement kinematic and dynamic control (PID, LQR, MPC) for precise motion.
- Use a camera and computer vision (OpenCV) for object recognition and pose estimation.
- Perform pick-and-place tasks autonomously.
- **Facilitate a smooth transition to physical hardware implementation (in a later phase, e.g., TCC2).**

## Components

- **4x NEMA 17 stepper motors**
- **1x MG996R servo motor** (gripper)
- **Drivers:** TB6600 and TMC2209
- **24V power supply**
- **Buck converter** (step down)
- **Raspberry Pi** (control and processing)
- **Logitech C920 USB Camera**
- **3D-printed structure (PLA)**

## Technologies and Tools

- **FreeCAD** – 3D modeling (for mechanical design)
- **Ultimaker Cura** – 3D printing preparation
- **KiCAD** – Circuit and PCB design
- **ROS2** – Robot Operating System (for modularity, communication, and control)
- **Gazebo** – Robotic simulation environment
- **URDF/XACRO** – Robot description format (for defining the robot in simulation)
- **OpenCV** – Computer vision library
- **C++ + Qt** – Desktop interface (for monitoring and control)

## Development Workflow

This project adopts an iterative, simulation-driven development workflow:

1.  **Mechanical Design (FreeCAD):** Detailed 3D modeling of the robotic arm's components and assembly.
2.  **Robot Description (URDF/XACRO):** Conversion of 3D models into URDF/XACRO format to define the robot's kinematics, dynamics, and visual properties for simulation.
3.  **Simulation Environment Setup (Gazebo):** Integration of the URDF robot model into Gazebo, configured for realistic physics simulation and visualization.
4.  **ROS2 Control & Vision Integration (Simulation Phase):**
    * Development and testing of kinematic and dynamic control algorithms (PID, LQR, MPC) within the simulated environment.
    * Implementation and validation of computer vision algorithms (OpenCV) using simulated camera feeds to identify and estimate object poses.
    * Integration of vision feedback into the control loops for autonomous pick-and-place tasks in Gazebo.
5.  **Physical Hardware Implementation (Future Phase):** Transitioning validated control and vision logic from simulation to the actual Raspberry Pi and robotic arm hardware, focusing on addressing real-world challenges such as sensor noise, motor calibration, and power management.

## Modeling and Control

- Direct and inverse kinematics using Denavit-Hartenberg parameters
- PID, LQR, and MPC control strategies
- Path planning based on visual input

## Computer Vision

- Color segmentation
- Contour detection and shape recognition
- Pose estimation
