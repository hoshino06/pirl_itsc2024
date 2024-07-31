#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import queue
import time
import cv2 # for camera

sys.path.append(os.pardir)
sys.path.append('../../')
from rl_env.carla_env   import CarEnv

#################################################
# Settings 
#################################################
carla_port = 5000
time_step  = 0.05 
map_train  = "../../maps/train.xodr"
spec_view   = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 
camera_view = {'x':-960, 'y':200, 'z':18, 'pitch':-70, 'yaw':130, 'roll':0} 
#camera_view = {'x':-970, 'y':195, 'z':15, 'pitch':-55, 'yaw':120, 'roll':0} 
IM_WIDTH, IM_HEIGHT = 640*3, 480*3

image_path = "trajectory_mapC_camera.png"

#################################################
# Load data 
#################################################
log_dir = 'data_trained/'
files   = os.listdir(log_dir)
files.sort()
file    = files[0]
data = np.load(log_dir+file)    

x   = data['position'][0,:] 
y   = data['position'][1,:] 
yaw = data['position'][2,:]

def spawn_vehicle(env, x, y, yaw):
    
    sp = env.all_sp[0]
    bp = env.vehicle_bp
    
    sp.location.x = x
    sp.location.y = y
    sp.rotation.yaw = yaw
    
    vehicle = env.world.spawn_actor(bp, sp)
    env.actor_list.append(vehicle)
    
def spawn_camera(env, camera_view): 

    sp = env.all_sp[0]
    sp.location.x = camera_view['x']
    sp.location.y = camera_view['y']
    sp.location.z = camera_view['z'] 
    sp.rotation.pitch = camera_view['pitch']
    sp.rotation.yaw   = camera_view['yaw']
    sp.rotation.roll   = camera_view['roll']
        
    cam_bp = env.bl.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov', '90')
    sensor = env.world.spawn_actor(cam_bp, sp)
    image_queue = queue.Queue()
    sensor.listen(image_queue.put)
    env.actor_list.append(sensor)
    
    return image_queue

################################################
# CarEnv
###############################################
try:
    env    = CarEnv(port=carla_port, time_step=time_step,
                 #autopilot       = True, #run in asynchronous mode 
                 custom_map_path = map_train,
                 actor_filter    = 'vehicle.audi.tt',  
                 spectator_init  = spec_view, 
                 spectator_reset = False, 
                 )

    # vehicles
    for t in np.arange(0, 5,0.5-1e-3): #[0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0-0.05]:
        i = int(t/time_step)
        spawn_vehicle(env, x[i], y[i], yaw[i])
        time.sleep(1)

    # camera 
    image_queue = spawn_camera(env, spec_view)

    env.world.tick()
    env.world.tick()
    
    image = image_queue.get()            
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("camera", i3)
    cv2.waitKey(1)
    cv2.imwrite(image_path, i3)

    
finally:
    if 'env' in locals():
        env.destroy()
        cv2.destroyAllWindows()