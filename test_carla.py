#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run carla simulator by
 ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=2000 &
"""


from pirl_carla.rl_env.carla_env    import CarEnv

try:
    
    carla_port = 2000
    time_step  = 0.05 
    
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}  #spectator coordinate  

    def choose_spawn_point(carla_env):
        sp_list = carla_env.get_all_spawn_points()    
        spawn_point = sp_list[1]
        return spawn_point

    env    = CarEnv(port=carla_port, time_step=time_step,
                    autopilot       = True, 
                    custom_map_path = None,
                    actor_filter    = 'vehicle.audi.tt',                      
                    spectator_init  = spec_town2,  
                    spectator_reset = False,
                    spawn_method    = choose_spawn_point,
                    )
    
    # Run simulation
    while True:

        _, _, isDone = env.step()
        velocity         = env.vehicle.get_velocity()
        angular_velocity = env.vehicle.get_angular_velocity()
        transform        = env.vehicle.get_transform()
        print(f"rotation: {transform.rotation}, velocity: {velocity}, angular velocity: {angular_velocity}\n")

        if isDone == True:
            env.reset()
                
except KeyboardInterrupt:
    print('\nCancelled by user - test_carla.py.')

finally:
    print('destroying actors and windows')
    if 'env' in locals():
        env.destroy()
    
    print('done.')

