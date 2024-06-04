"""
Plot safe probability vs map
"""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam

path_to_carla = os.path.expanduser("~/carla/carla_0_9_15")
sys.path.append(glob.glob(path_to_carla + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
import carla

###########################################################################
# Load PIRL agent and carla environment     

sys.path.append(os.pardir)
sys.path.append('.')
from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions
from training_pirl_MapC import Env, convection_model, diffusion_model, sample_for_pinn, map_c_before_corner

def contour_plot(x, y, z, key = None, filename=None):
    #(x, y) = (e, psi)/ (e, vx)
    x, y = np.meshgrid(x, y)
    x = x.transpose()
    y = y.transpose()
    
    # Creating the contour plot
    plt.figure(figsize=(6, 4))
    contour = plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(contour)  # Adding color bar to show z values
    
    # Adding labels and title
    if not key:
        plt.xlabel(r'Slip anlge $\beta$ [deg]')
        plt.ylabel('Yaw rate $r$ [deg/s]')       
    elif key[0] == "e" and key[1] == 'psi':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel(r'Heading error $\psi$ [rad]')
    elif key[0] == "e" and key[1] == 'v':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel('Longitudinal velocity $v_x$ [m/s]')
    
    if filename:
       
       plt.savefig(filename, bbox_inches="tight")
    
    # Displaying the plot
    plt.show()

def heatmap_plot(x, y, z, key = ["e", "psi"]):
    #(x, y) = (e, psi)/ (e, vx)
    """
    x, y = np.meshgrid(x, y)
    
    # Creating the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(contour)  # Adding color bar to show z values
    """
    # Adding labels and title
    plt.xlabel(key[0])
    plt.ylabel(key[1])
    plt.title('Safety Probability')
    plt.imshow(z)
    
    # Displaying the plot
    plt.show()
    

###############################################################################
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    

    ###########################
    # Get nominal state
    carla_port = 4000
    time_step  = 0.05    
    map_train  = "/home/ubuntu-root/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/train.xodr"
    spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

    try:  
        # Get reference state
        rl_env = Env(port=carla_port, 
                     time_step=time_step, 
                     custom_map_path = map_train,
                     actor_filter    = 'vehicle.audi.tt',
                     spawn_method    = map_c_before_corner,
                     spectator_init  = spec_mapC_NorthEast, 
                     waypoint_itvl   = 3.0,
                     spectator_reset = False, 
                     autopilot       = False)
        rl_env.reset()

        x_vehicle = np.array( rl_env.getVehicleState() )
        x_road    = np.array( rl_env.getRoadState() )

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_vehicle: {x_road}')

        actNum = rl_env.action_num
        obsNum = len(rl_env.reset())

    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()


    ##############################
    # Load agent
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, actNum),
                nn.Sigmoid()
                )
        def forward(self, x):
            output = self.linear_stack(x)
            return output    
    model = NeuralNetwork().to('cpu')

    
    agentOp = agentOptions(
        DISCOUNT   = 1, 
        #OPTIMIZER  = Adam(learning_rate=1e-4),
        REPLAY_MEMORY_SIZE = 5000, 
        REPLAY_MEMORY_MIN  = 1000,
        MINIBATCH_SIZE     = 32,
        EPSILON_INIT        = 1, #0.9998**20_000, 
        EPSILON_DECAY       = 0.9998, 
        EPSILON_MIN         = 0.01,
        )
    
    pinnOp = pinnOptions(
        CONVECTION_MODEL = convection_model,
        DIFFUSION_MODEL  = diffusion_model,   
        SAMPLING_FUN     = sample_for_pinn,
        WEIGHT_PDE       = 1e-4, 
        WEIGHT_BOUNDARY  = 1, 
        HESSIAN_CALC     = False,
        )    
    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    #agent.load_weights('ITSC2024data/MapC/hoshino/04071140-19k', ckpt_idx='latest')
    agent.load_weights('logs/MapC/04251704', ckpt_idx='latest')

    ###########################################
    # Safety probability    
    rslu = 40
    horizon   = 5
    velocity  = 30
    beta_list = np.linspace(-50, 0, rslu)
    yawr_list = np.linspace(30, 90, rslu)
    
    safe_p = np.zeros((len(beta_list), len(yawr_list)))
    for i in range(len(beta_list)):
        #print(f'{i}/{len(beta_list)}')
        x_vehicle[0] = velocity
        x_vehicle[1] = beta_list[i]
        for j in range(len(yawr_list)):
            x_vehicle[2] = yawr_list[j]
            s            = np.concatenate([ x_vehicle, x_road, np.array([horizon]) ])       
            safe_p[i][j]   = agent.get_qs(s).max()

    np.savez("plot/MapC/safe_prob.npz", beta=beta_list, yawr=yawr_list, prob=safe_p)
     
    contour_plot(x=beta_list, y=yawr_list, z=safe_p,
                 filename='plot/MapC/mapC_safe_prob.png')
    #heatmap_plot(x=beta_list, y=yawr_list, z=safe_p, key=['slip angle', 'yaw rate'])
