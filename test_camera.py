# -*- coding: utf-8 -*-
## Run CARLA Server first
# $ ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &

# Load standard libralies
import glob
import os
import sys

import numpy as np
import cv2 # openCV for camera
import queue

# Load carla module
path_to_carla = os.path.expanduser("~/carla/carla_0_9_15")

sys.path.append(glob.glob(path_to_carla + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

import carla

# Run test code with python API

actor_list = []
try:
    # Client creation and world connection
    client = carla.Client('localhost',3000)
    client.set_timeout(10.0) # seconds
    world = client.get_world()

    # Set map to Town02
    if not world.get_map().name == 'Carla/Maps/Town02':
        world = client.load_world('Town02')

    # Set synchronous mode settings
    # new_settings = world.get_settings()
    # new_settings.synchronous_mode = True
    # new_settings.fixed_delta_seconds = 0.1
    # world.apply_settings(new_settings)
        
    # Create vehicle actor
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[1]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    actor_list.append(vehicle)

    # Create camera actor
    IM_WIDTH = 640
    IM_HEIGHT = 480
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov', '110')
    spawn_point = carla.Transform(carla.Location(x=2.5, z=10.0), carla.Rotation(pitch=-90))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    image_queue = queue.Queue()
    sensor.listen(image_queue.put)
    actor_list.append(sensor)

    # Func: camera processing by opencv
    def process_img(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)    
        return i3/255.0
    
    # Run simulation
    while True:

        world.tick()
        velocity         = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()
        transform = vehicle.get_transform()

        print(transform.location)
        #print(f"rotation: {transform.rotation}, velocity: {velocity}, angular velocity: {angular_velocity}")

        image = image_queue.get()
        process_img(image)
        
except KeyboardInterrupt:
    print('\nCancelled by user.')

finally:
    print('destroying actors and windows')
    for actor in actor_list:
        actor.destroy()

    print('done.')

    
