# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 05:37:43 2023

@author: hoshino
"""
import glob
import os
import sys
import numpy as np

import queue
import cv2 # for camera
import math

# Load carla module
path_to_carla = os.path.expanduser("~/carla/carla_0_9_15")

sys.path.append(glob.glob(path_to_carla + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

import carla


def random_spawn_point_corner(spawn_point, start: "dict", corner: "dict", end: "dict"):
    dist_1 = abs(corner['location']['y']-start['location']['y'])
    dist_2 = abs(end['location']['x']-corner['location']['x'])
    distance = dist_1 + dist_2
    eps = np.random.rand()
    rand_dist = eps*distance
    if rand_dist > dist_1:
        # 2nd phase, moving in x direction
        spawn_point = carla.Transform(carla.Location((end['location']['x']-(distance-rand_dist)), end['location']['y'], end['location']['z']), carla.Rotation(end['rotation']['pitch'], end['rotation']['yaw'], end['rotation']['roll']))
    else:
        # 1nd phase, moving in y direction
        spawn_point = carla.Transform(carla.Location(start['location']['x'], start['location']['y']+rand_dist, start['location']['z']), spawn_point.rotation)
    return spawn_point


    
def load_custom_map(xodr_path, client):
    if os.path.exists(xodr_path):
        with open(xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be readed.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0  # in meters
        max_road_length = 500.0 # in meters
        wall_height = 1.0      # in meters
        extra_width = 0.6      # in meters
        world = client.generate_opendrive_world(
            data, carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                max_road_length=max_road_length,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=True,
                enable_mesh_visibility=True))
    else:
        print(os.getcwd())
        print('file not found.')
    return world

def fetch_all_spawn_point(world):
    map = world.get_map()
    all_lanes = map.get_topology()
    all_wp_list = []
    for lane in all_lanes:
        start_wp = lane[0]
        all_wp_list += start_wp.next_until_lane_end(1.0)
    all_sp_transform_list = []
    for wp in all_wp_list:
        spawn_point = wp.transform
        spawn_point.location.z = 0.500
        all_sp_transform_list.append(spawn_point)
    return all_sp_transform_list
    

def random_spawn_custom_map(all_spawn_point_list):
    rand_1 = np.random.randint(0,len(all_spawn_point_list))
    return all_spawn_point_list[rand_1]

# CarEnv Class
class CarEnv:

    step_T_pool = np.linspace(0.6,1, num=5)    # throttle values
    step_S_pool = np.linspace(-0.8,0.8, num=5) # steering angle values
    action_num  = len(step_T_pool) * len(step_S_pool)

    def __init__(self, port=2000, time_step   = 0.01, autopilot=False, custom_map_path = None):
        """
        Connect to Carla server, spawn vehicle, and initialize variables 
        """
        # Client creation and world connection
        print(f'Connecting carla server on port {port}.')
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.start_point = None
        self.custom_map_path = custom_map_path
        self.all_sp = None
        # load custom map
        if self.custom_map_path != None:
            self.world = load_custom_map(self.custom_map_path, self.client)
            self.all_sp = fetch_all_spawn_point(self.world)
        else:
            #if not self.world.get_map().name == 'Carla/Maps/Town02':
            self.world = self.client.load_world('Town02')
            #self.world = self.client.generate_opendrive_world(opendrive = "Carla/Maps/Town02")
            
            #print(f"Map Layer:{dir(carla.MapLayer)}")
            #exit()
                
        # Set synchronous mode settings
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = time_step
        self.time_step = time_step
        self.world.apply_settings(new_settings)

        # Initialization
        self.actor_list = []

        # Create vehicle actor
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]
        new_spawn_point = self.world.get_map().get_spawn_points()[1]
        self.vehicle = self.world.spawn_actor(bp, new_spawn_point)

        self.actor_list.append(self.vehicle)
        if autopilot: 
            self.vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
            print('Vehicle was spawned with auto pilot mode')
        self.autopilot = autopilot
        
        # Create camera actor
        self.IM_WIDTH, self.IM_HEIGHT = 640, 480
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', f'{self.IM_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{self.IM_HEIGHT}')
        cam_bp.set_attribute('fov', '110')
        spawn_point = carla.Transform(carla.Location(x=-7.390556, y=312.114441, z=10.220332), carla.Rotation(pitch=-20, yaw=-45))
        sensor = self.world.spawn_actor(cam_bp, spawn_point)
        self.image_queue = queue.Queue()
        sensor.listen(self.image_queue.put)
        self.actor_list.append(sensor)  

        # Run one step and store state dimension
        initState = self.reset()
        self.state_dim  = len(initState)

    def generate_random_spawn_point(self):
        world_map = self.world.get_map()
        new_spawn_point = world_map.get_spawn_points()[1]
        start_point = {'location':{'x':-7.530000, 'y':270.729980, 'z':0.500000}, 'rotation':{'pitch':0.000000,'yaw':89.99954,'roll':0.000000}}
        corner_point = {'location':{'x':-7.390556, 'y':303.114441, 'z':0.520332}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
        end_point = {'location':{'x':25.694059, 'y':306.545349, 'z':0.521810},'rotation':{'pitch':0.000000,'yaw':0.000,'roll':0.000000}}
        if self.custom_map_path != None:
            spawn_point = random_spawn_custom_map(self.all_sp)
            #self.vehicle.set_transform(spawn_point)
            self.start_point = spawn_point
            #print(spawn_point.location)
        else:
            
            spawn_point_trans = random_spawn_point_corner(new_spawn_point,start_point, corner_point, end_point)
            way_point = world_map.get_waypoint(spawn_point_trans.location, project_to_road=True)
            x_rd   = way_point.transform.location.x
            y_rd   = way_point.transform.location.y
            yaw_rd = way_point.transform.rotation.yaw
            spawn_point_trans = carla.Transform(carla.Location(x_rd, y_rd,0.50000), 
                                          carla.Rotation(0, yaw_rd, 0))  
            self.start_point = spawn_point_trans 
            
        return
    
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)    
        return i3/255.0
    

    def reset(self):
        """
        Initialize vehicle state
        """
        
        # Initialization of vehicle position and heading angle
        x_loc    = 0
        y_loc    = 0.5
        psi_loc  = 0
        #start_rd = self.world.get_map().get_spawn_points()[1]
        self.generate_random_spawn_point()
        start_rd = self.start_point
        rotation = self.start_point.rotation

        yaw_st   = start_rd.rotation.yaw
        x_wld, y_wld = self.local2world(x_loc, y_loc, yaw_st)
        location = start_rd.location + carla.Location(x=x_wld, y=y_wld, z=0)
        rotation = carla.Rotation( yaw = yaw_st + psi_loc )
        trans = carla.Transform(location, rotation)
        self.vehicle.set_transform(trans)
        
        
                
        # Initialization of vehicle vlocity
        vx = 10
        vy = 0
        world_vx, world_vy = self.local2world(vx, vy, rotation.yaw)        
        velocity_world = carla.Vector3D(world_vx, world_vy, 0)
        self.vehicle.set_target_velocity(velocity_world) # effective after two frames
        
        # Initialization of vechicle angular velocity
        yaw_rate = 0
        angular_velocity = carla.Vector3D(z = yaw_rate)
        self.vehicle.set_target_angular_velocity(angular_velocity)

        # Get initial state
        x_vehicle = self.getVehicleState()
        x_road    = self.getRoadState()            
        initState = x_vehicle + x_road
        return initState
                

    def step(self, actionID=0):
        """
        Apply control and return new state, reward, and isDone flag
        """

		# apply the control commands
        throttleID = int(actionID / len(self.step_S_pool))
        steerID = int(actionID % len(self.step_S_pool))

        self.control = carla.VehicleControl(
								throttle = self.step_T_pool[throttleID],
								steer = self.step_S_pool[steerID],
								brake = 0.0,
								hand_brake = False,
								reverse = False,
								manual_gear_shift = False,
								gear = 0)
        
        if not self.autopilot: 
            self.vehicle.apply_control(self.control)

        # get new state
        self.world.tick()
        x_vehicle = self.getVehicleState()
        x_road    = self.getRoadState()            
        newState = x_vehicle + x_road

        # Is done?
        location = self.vehicle.get_location()
        lat_err  = x_road[0]        
        if location.x > 20: # Goal
            isDone = True
            reward = 1
            
        elif np.abs( lat_err ) > 1:  # Unsafe
            isDone = True
            reward = 0
        else:
            isDone  = False
            reward  = 0

        return newState, reward, isDone


    def getVehicleState(self):
        
        # Get velocity
        velocity_world = self.vehicle.get_velocity()
        vx_wld = velocity_world.x
        vy_wld = velocity_world.y
        yaw    = self.vehicle.get_transform().rotation.yaw
        local_vx, local_vy = self.world2local(vx_wld, vy_wld, yaw)

        # Get slip angle 
        if local_vx > 1e-5:
            slip_angle = np.arctan(local_vy/local_vx)/3.1415926*180
        else:
            slip_angle = 0
        
        # Get yaw rate
        angular_velocity = self.vehicle.get_angular_velocity()
        yaw_rate = angular_velocity.z
                
        # Vehicle state 
        x_vehicle = [local_vx, slip_angle, yaw_rate] 
        
        return x_vehicle

    def fetch_relative_states(self, world_map, wp_transform, interval, next_num)->(list, list):
        relative_x = []
        relative_y = []
        wp_transform_list = [wp_transform]
        x = wp_transform.location.x
        y = wp_transform.location.y
        for i in range(5):
            yaw = wp_transform.rotation.yaw
            cur_x = wp_transform.location.x
            cur_y = wp_transform.location.y
            cur_z = wp_transform.location.z
            #print(cur_x, cur_y)
            x_dis = cur_x + math.cos(yaw)*interval
            y_dis = cur_y + math.sin(yaw)*interval
            waypoint = world_map.get_waypoint(carla.Location(x_dis, y_dis, cur_z),project_to_road=True, lane_type=(carla.LaneType.Driving))
            wp_transform = waypoint.transform
            wp_transform_list.append(wp_transform)
            relative_x.append(wp_transform.location.x-x)
            relative_y.append(wp_transform.location.y-y)
        result = relative_x + relative_y
        return result, wp_transform_list
        
    def getRoadState(self):
        
        # Get vehicle location
        vehicle_trans = self.vehicle.get_transform()  
        vehicle_locat = vehicle_trans.location
        vehicle_rotat = vehicle_trans.rotation 
        x   = vehicle_locat.x
        y   = vehicle_locat.y
        yaw = vehicle_rotat.yaw

        # Get nearby waypoint
        world_map = self.world.get_map()
        
        way_point = world_map.get_waypoint(vehicle_locat, project_to_road=True)
        wp_transform = way_point.transform
        interval = 0.5
        next_num = 5
        vector_list, future_wp_list = self.fetch_relative_states(world_map, wp_transform, interval, next_num)
        x_rd   = way_point.transform.location.x
        y_rd   = way_point.transform.location.y
        yaw_rd = way_point.transform.rotation.yaw
        #road_state = []
        #for wp_t in future_wp_list:
        loc_x, loc_y  = self.coordinate_w2l(vehicle_locat, way_point.transform)

        # Error variables
        
        #e   = np.linalg.norm( [x - x_rd, y - y_rd] )
        psi = yaw - yaw_rd 
        if psi < -180:
            psi += 360
        if psi > 180:
            psi -= 360
        psi = psi / 180.0 * 3.141592653	
        
        # print(f'veh=[{x:.1f},{y:.1f}], rd=[{x_rd:.1f},{y_rd:.1f}], e={e:.1f}')
        # print(f'yaw={yaw_rd:.1f}, yaw_rd={yaw_rd:.1f}, psi={psi:.1f}')
        # print(f'e={e:.1f}, psi={psi:.1f}')

        #return [e, psi, tuple(relative_x), tuple(relative_y)]
        return [loc_x, psi] + vector_list


    def local2world(self, long, lat, yaw):
        
        # local x and y
        x = long
        y = lat
        
        # transformation
        yaw = yaw / 180.0 * 3.141592653
        world_x   = x * np.cos(yaw) - y * np.sin(yaw)
        world_y   = y * np.cos(yaw) + x * np.sin(yaw)
        
        return world_x, world_y 


    def world2local(self, world_x, world_y, yaw):

        # world x and y
        x = world_x
        y = world_y 
        
        # transformation
        yaw = yaw / 180.0 * 3.141592653
        loc_x   = x * np.cos(-yaw) - y * np.sin(-yaw)
        loc_y   = y * np.cos(-yaw) + x * np.sin(-yaw)
        
        return loc_x, loc_y
    
    def coordinate_w2l(self, v_location, w_transform):
        x = w_transform.location.x - v_location.x
        y = w_transform.location.y - v_location.y
        theta = (90 - w_transform.rotation.yaw)/ 180.0 * 3.141592653
        # theta is the rotate of relative vector between waypoint and vehicle location
        loc_x   = x * np.cos(theta) - y * np.sin(theta)
        loc_y   = y * np.cos(theta) + x * np.sin(theta)
        return loc_x, loc_y


    def destroy(self):
        
        # Destroy actors
        print('destroying actors and windows')
        for actor in self.actor_list:
            actor.destroy()
        self.world.tick()
        print('done.')
              

##############################################################################
# main
##############################################################################
if __name__ == '__main__': 
   
    carla_port = 6000
    time_step  = 0.01
    
    try:
        rl_env = CarEnv(port= carla_port, 
                        time_step=time_step) 
                        #custom_map_path="/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test.xodr") #, autopilot=True)
        while True:
            rl_env.reset()
            isDone = False
            while not isDone:    
                 newState, reward, isDone = rl_env.step()    
    
    except KeyboardInterrupt:
        print('\nCancelled by user.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()

