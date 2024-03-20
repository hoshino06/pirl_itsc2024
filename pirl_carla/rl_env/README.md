# rl_env.carla_env.CarEnv
## Methods
- __init__(self, 
                 port=2000, time_step = 0.01, autopilot=False, 
                 custom_map_path = None, 
                 actor_filter    = 'vehicle.audi.tt', # or use 'model3' 
                 spawn_method    = None,
                 vehicle_reset   = None,
                 spectator_reset = True, 
                 camera_view     = None, 
                 initial_speed   = 10)
  - 'port =2000': Port number of carla simulator
  - 'time_step = 0.01': Time step of carla simulation
  - 'autopilot = False':  
  - 'custom_map_path = None': Specify path to the custom map. If None, Town2 will be loaded.             
  - 'actor_filter = 'vehicle.audi.tt'':  
  - 'spawn_method = None': Specify the method of spaning vehicle. 
  - 'vehicle_reset = None': 
  - 'spectator_init  = None': Spector view can be specifed by dictionaly  
  - 'spectator_reset = True': 
  - 'camera_view     = None': 
  - 'initial_speed   = 10': 




