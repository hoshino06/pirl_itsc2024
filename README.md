# Autonomous Drifting Based on Maximal Safety Probability Learning
*27th IEEE International Conference on Intelligent Transportation Systems (ITSC 2024)*


## Requirements
1. Tested on Ubuntu 22.04.4 LTS
2. Nvidia GPU equipped, and driver Installed. Tested on GeForce RTX 3080. 
3. Install [CARLA simulator](https://carla.org/), which is an open-source simulator for autonomous driving research. Tested on CARLA 0.9.15. 
4. Install [Anaconda](https://www.anaconda.com/), which is a package manager, environment manager, and Python distribution.
5. Setup conda environment
   ```console
    conda env create -f environment_pirl_carla.yaml
   ```

## Start and Test Simulator

1. We tested codes on CARLA installed via [Package installation](https://carla.readthedocs.io/en/0.9.15/start_quickstart/#b-package-installation)
2. After the installation, run CARLA while specifying the port (assume simulator is installed in `~carla/calra_0_9_15/`):
   ```console
    ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=2000 &
   ```
3. Run test code to make sure python API is available (if it fails, see step 4):
   ```console
    python test_carla.py 
   ```
4. Rewrite the path to carla API at line 18 of `pirl_carla/rl_env/carla_env.py`
   ```python
    path_to_carla = os.path.expanduser("~/carla/carla_0_9_15")
   ```

## Lane keeping with normal cornering

<div align=left> 
<img src="./pirl_carla/plot/Town2/simulation.gif" width=300 alt="Normal cornering"/>
</div>

- Run training script:
  ```console
  ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=2000 &
  cd pirl_carla
  conda activate pirl_carla
  python training_pirl_Town2.py
  tensorboard --logdir logs/
  ```
  
- Run verification script:
  ```console
   ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
  conda activate pirl_carla
  python verification_Town2.py
  ```

## Safe drifting

Racing circuit is used for demonstrating safe drifting. Custom map developed in [drift_drl](https://github.com/caipeide/drift_drl) is used. 


- Run training script:
  ```console
  ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=4000 &
  python training_pirl_MapC.py
  tensorboard --logdir logs/
  ```
- Run verification scrpit:
  ```console
  ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
  python verification_MapC.py
  ```

  
