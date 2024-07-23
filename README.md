# Autonomous Drifting Based on Maximal Safety Probability Learning
*27th IEEE International Conference on Intelligent Transportation Systems (ITSC 2024)*


## Requirements
1. Tested on Ubuntu 22.04.4 LTS
2. Nvidia GPU equipped, and driver Installed. Tested on GeForce RTX 3080. 
3. Install [CARLA simulator](https://carla.org/), which is an open-source simulator for autonomous driving research. Tested on CARLA 0.9.15. 
4. Install [Anaconda](https://www.anaconda.com/), which is a package manager, environment manager, and Python distribution.
5. Setup conda environment


## Start and Test Simulator

- We tested codes on CARLA installed via (Package installation)[https://carla.readthedocs.io/en/0.9.15/start_quickstart/#b-package-installation]
- After the installation, run CARLA while specifying the port (assume simulator is installed in `~carla/calra_0_9_15/`):
```console
 ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
```
- Run test code to make sure python API is available:
```console
 python test_carla.py 
```

## Lane keeping with normal cornering



## Safe drifting

