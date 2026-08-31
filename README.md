# GICP Mapper

ROS 2 - Humble package for point cloud registration and volumetric map generation using [Fast_GICP](https://github.com/koide3/fast_gicp) and [Voxeland](https://github.com/MAPIRlab/Voxeland).



## Overview

GICP Mapper is a ROS 2 - Humble package for point cloud registration and volumetric mapping. The package uses GICP/VGICP/NDT/NDT-D2D algorithms to estimate the relative transformation between consecutive point clouds and incrementally build a consistent 3D map.

The mapping process is based on [Voxeland](https://github.com/MAPIRlab/Voxeland), providing an efficient voxel-based representation of the environment. The package is designed to provide a modular and configurable framework for experimenting with different registration and mapping configurations.

It also includes a node based on [Bonxai](https://github.com/facontidavide/Bonxai) that implements a probabilistic occupancy grid map server, storing the point distribution within each voxel.

## Features

This repo includes
- Generalized Iterative Closest Point 
- Voxelized - Generalized Iterative Closest Point
- Normal Distribution Transform
- NDT - D2D and Probabilistic Map server Node

## Installation

Install dependencies, alongside [Voxeland](https://github.com/MAPIRlab/Voxeland) and [Fast_GICP](https://github.com/koide3/fast_gicp), clone this repo and you’re ready to go.

```bash
git clone https://github.com/IsraelNunezDeArenas/gicp_mapper.git
```
### Requirements

- Ubuntu 22.04
- ROS2 - Humble
- Ubuntu 22.04
- CMake ≥ 3.8
- C++20 compiler
- OpenMP
- [Bonxai](https://github.com/facontidavide/Bonxai), [Voxeland](https://github.com/MAPIRlab/Voxeland), [Fast_GICP](https://github.com/koide3/fast_gicp).

