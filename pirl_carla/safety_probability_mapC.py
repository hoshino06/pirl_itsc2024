"""
Plot safe probability vs map
"""
#import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

sys.path.append(os.pardir)
sys.path.append('.')
from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions
from training_pirl_MapC import Env, convection_model, diffusion_model, sample_for_pinn, map_c_before_corner

###########################################################
# Settings
###########################################################
log_dir = 'logs/MapC/07282145'

carla_port = 5000
time_step  = 0.05
map_train  = "./maps/train.xodr"
spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

#############################################################
# Load PIRL agent and carla environment     
############################################################

def load_agent(env, log_dir):

    actNum = env.action_num
    obsNum = len(env.reset())

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
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
    
    agentOp = agentOptions()
    pinnOp = pinnOptions(convection_model,diffusion_model,sample_for_pinn)    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    agent.load_weights(log_dir, ckpt_idx='latest')

    return agent


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
    
    plt.show()
    

###############################################################################
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    

    ###################################
    # Load env and agent
    ##################################
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
        agent = load_agent(rl_env, log_dir)

        x_vehicle = np.array( rl_env.getVehicleState() )
        x_road    = np.array( rl_env.getRoadState() )

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_vehicle: {x_road}')

    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()

    #####################################
    # Safety probability  
    #####################################
    
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
    
    contour_plot(x=beta_list, y=yawr_list, z=safe_p,
                 filename='plot/MapC/mapC_safe_prob.png')
