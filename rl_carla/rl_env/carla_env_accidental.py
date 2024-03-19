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
import time

# Load carla module
path_to_carla = os.path.expanduser("~/carla/carla_0_9_15")

sys.path.append(glob.glob(path_to_carla + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

import carla

############################################
# Load custom map (racing track)    
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
        raise FileNotFoundError('Custom map file not found')
    return world

def draw_path(location, world, life_time=1.0, string = None):
    if string == None:
        world.debug.draw_string(location, 'O', draw_shadow=False,
                                color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                persistent_lines=True)
    elif string != None:
        world.debug.draw_string(location, string, draw_shadow=False,
                                color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                persistent_lines=True)
     
def fetch_all_spawn_point(world):
    map = world.get_map()
    all_lanes = map.get_topology()
    all_wp_list = []
    for lane in all_lanes:
        start_wp = lane[0]
        if start_wp.lane_width > 20:
            wp_list = start_wp.next_until_lane_end(1.0)
            all_wp_list += wp_list
            #for sp in wp_list:
            #    draw_path(sp.transform.location, world, life_time= 600)
            #set_spectator(world, wp_list[0].transform)
            #print(start_wp.road_id, start_wp.lane_id, start_wp.section_id, start_wp.lane_width)
        else:
            continue
    #for sp in all_wp_list:
    #    draw_path(sp.transform.location, world, life_time= 600)    
    #set_spectator(world, all_wp_list[0].transform)
    
    all_sp_transform_list = []
    for wp in all_wp_list:
        spawn_point = wp.transform
        spawn_point.location.z = 0.500
        all_sp_transform_list.append(spawn_point)
    return all_sp_transform_list

########################################
# Get spawn point for Town 2 
def random_spawn_point_corner(spawn_point, start: "dict", corner: "dict", end: "dict"): # todo, wrap if else in bigger if else checking if cutsom_map
    dist_1 = abs(corner['location']['y']-start['location']['y'])
    dist_2 = abs(end['location']['x']-corner['location']['x'])
    distance = dist_1 + dist_2
    eps = np.random.rand()
    rand_dist = eps*distance
    if rand_dist > dist_1:
        # 2nd phase, moving in x direction
        new_spawn_point = carla.Transform(carla.Location((end['location']['x']-(distance-rand_dist)), end['location']['y'], end['location']['z']), carla.Rotation(end['rotation']['pitch'], end['rotation']['yaw'], end['rotation']['roll']))
    else:
        # 1nd phase, moving in y direction
        new_spawn_point = carla.Transform(carla.Location(start['location']['x'], start['location']['y']+rand_dist, start['location']['z']), spawn_point.rotation)
    return spawn_point

def random_spawn_point_corner_new_map(spawn_point, start, corner, end):
    dist_1 = abs(corner['location']['x']-start['location']['x'])
    dist_2 = abs(end['location']['y']-corner['location']['y'])
    distance = dist_1 + dist_2
    eps = np.random.rand()
    rand_dist = eps*distance
    if rand_dist < dist_1:
        # 2nd phase, moving in y direction
        new_spawn_point = carla.Transform(carla.Location(corner['location']['x'], corner['location']['y'], corner['location']['z']), carla.Rotation(end['rotation']['pitch'], end['rotation']['yaw'], end['rotation']['roll']))
    else:
        # 1nd phase, moving in x direction
        new_spawn_point = carla.Transform(carla.Location((start['location']['x']), start['location']['y'], start['location']['z']), spawn_point.rotation)
    return new_spawn_point

######################################
# Get spawn point for racing track
def random_spawn_custom_map(all_spawn_point_list):
    rand_1 = np.random.randint(0,len(all_spawn_point_list))
    '''for sp in all_spawn_point_list:
        print(sp.location)'''
    return all_spawn_point_list[rand_1]

def spectate(world):
    while(True):
        t = world.get_spectator().get_transform()
        #coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
        coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
        print (coordinate_str)
        time.sleep(100)
        
def set_spectator(world, transform):
    spectator = world.get_spectator()
    spectator.set_transform(transform)
    return


###############################################################################        
# CarEnv Class
class CarEnv:

    step_T_pool = np.linspace(0.6,1, num=5)    # throttle values
    step_S_pool = np.linspace(-0.8,0.8, num=5) # steering angle values
    action_num  = len(step_T_pool) * len(step_S_pool)

    def __init__(self, 
                 port=2000, time_step = 0.05, autopilot=False, 
                 custom_map_path = None, 
                 spawn_method = None, 
                 camera_view  = None,
                 initial_speed = 10):
        """
        Connect to Carla server, spawn vehicle, and initialize variables 
        """
        # Initialization of attributes
        self.actor_list  = []
        self.image_queue = None
        self.autopilot   = autopilot
        self.spawn_method = spawn_method
        self.camera_view  = camera_view
        self.initial_speed = initial_speed
        
        # Client creation and world connection
        print(f'Connecting carla server on port {port}.')
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.custom_map_path = custom_map_path
        self.all_sp = None

        # load custom map
        self.initial_speed = initial_speed
        if self.custom_map_path != None:
            self.world = load_custom_map(self.custom_map_path, self.client)
            self.all_sp = fetch_all_spawn_point(self.world)
            #    draw_path(sp.location, self.world, life_time= 600)
        else:
            if not self.world.get_map().name == 'Carla/Maps/Town02':
                self.world = self.client.load_world('Town02')
                
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
        self.bl = blueprint_library
        bp = blueprint_library.filter('model3')[0]
        #new_spawn_point = self.world.get_map().get_spawn_points()[1]
        new_spawn_point = carla.Transform(carla.Location(x=-980.518616, y=202.946793, z=0.500000))
        self.vehicle = self.world.spawn_actor(bp, new_spawn_point)
        
        self.actor_list.append(self.vehicle)
        if autopilot: 
            self.vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
            print('Vehicle was spawned with auto pilot mode')
        self.autopilot = autopilot
        
        # Change spectator view (carla window)
        sp_loc   = new_spawn_point.location
        sp_rot   = new_spawn_point.rotation
        spec_loc = sp_loc + carla.Location(x=0, y=0, z=2)         
        trans    = carla.Transform(spec_loc, sp_rot)
        self.world.get_spectator().set_transform(trans)        
        
        # Create camera actor (only for Town2)
        if custom_map_path == None:
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
        
    def display_spawn_points(self):
        print("inside function")
        for sp in self.all_sp:
            print(1)
            x_loc, y_loc, z_loc = sp.location.x, sp.location.y, sp.location.z
            p_rot, y_rot, r_rot = sp.rotation.pitch, sp.rotation.yaw, sp.rotation.roll
            
            draw_path(sp.location, self.world, life_time=600)
            
            # create vehicle actor
            sp_vehicle = sp
            veh_bp = self.bl.filter('model3')[0]
            vehicle = self.world.spawn_actor(veh_bp, sp_vehicle)
            
            # create camera actor
            bp_camera = self.bl.find('sensor.camera.rgb')
            bp_camera.set_attribute('image_size_x', '640')
            bp_camera.set_attribute('image_size_y', '480')
            bp_camera.set_attribute('fov', '110')
            sp_camera = carla.Transform(carla.Location(x=x_loc, y=y_loc, z=z_loc), carla.Rotation(pitch=p_rot, yaw=y_rot, roll=r_rot))
            sensor = self.world.spawn_actor(bp_camera, sp_camera)
            self.image_queue = queue.Queue()
            sensor.listen(self.image_queue.put)
            
            print(x_loc, y_loc, z_loc)
            print(sensor.get_location())
            print("--------------------------------")
            
            '''while True:
                if not self.image_queue.empty():  # Check if the queue has items waiting
                    print("if")
                    image = self.image_queue.get()
                    self.process_img(image)
                    break  # Exit the loop once an image is processed'''
            
            vehicle.destroy()
            sensor.destroy()
            
            '''# Retrieve image from the camera sensor
            image = self.image_queue.get()
            
            # Convert image to numpy array
            img = np.array(image.raw_data)
            
            # Reshape image to the correct dimensions and channels
            img = img.reshape((image.height, image.width, 4))
            
            # Convert RGBA to BGR (OpenCV uses BGR color format)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Display the image using OpenCVq
            #cv2.imshow("Camera POV", img)
            # process camera image data
            while True:
                if not self.image_queue.empty():  # Check if the queue has items waiting
                    print("if")
                    image = self.image_queue.get()
                    self.process_img(image)
                    break  # Exit the loop once an image is processed
                    
            #time.sleep(1)  # Add delay after processing each spawned actor

            
            # Wait for a key press and check if the user pressed the 'q' key to quit

            #self.actor_list.append(vehicle)
            #self.actor_list.append(sensor)'''
            
        # Destroy OpenCV window and cleanup
        cv2.destroyAllWindows()
            

    ###########################################################################
    def reset(self):
        """
        Initialize vehicle state
        """
        ##########
        # Vehicle position and heading angle (wrt spanw point)
        x_loc    = 0
        y_loc    = 0
        psi_loc  = np.random.uniform(-60, 60)
       # psi_loc = 0
        
        ##########
        # Choose spawn point and set camera
        #spawn_point = self.world.get_map().get_spawn_points()[1]
        spawn_point = self.generate_random_spawn_point()
        # , carla.Rotation(pitch=0, yaw=90, roll=0)
        
        '''start_point = {'location':{'x':-1005.017395, 'y':203.138901, 'z':0.485300}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
        corner_point = {'location':{'x':-967.017395, 'y':203.138901, 'z':0.485300}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
        end_point = {'location':{'x':-967.017395, 'y':230.138901, 'z':0.485300},'rotation':{'pitch':0.000000,'yaw':89.99954,'roll':0.000000}}'''

        #spawn_point = carla.Transform(carla.Location(x=-999.017395, y=203.138901, z=0.485300)) # og start pos
        #cam_spawn_point = carla.Transform(carla.Location(x=-965.017395, y=185.138901, z=0.485300), carla.Rotation(yaw=-90))
        
        #spawn_point = carla.Transform(carla.Location(x=-980.518616, y=202.946793, z=0.500000))
        cam_spawn_point = carla.Transform(carla.Location(x=-965.017395, y=185.138901, z=1.485300), carla.Rotation(pitch=-45, yaw=120))
        if self.custom_map_path != None:
            self.set_camera(self.bl, cam_spawn_point)
            
        ##########
        # Change spectator view (carla window)
        sp_loc   = spawn_point.location
        sp_rot   = spawn_point.rotation
        spec_loc = sp_loc + carla.Location(x=0, y=0, z=5)         
        trans    = carla.Transform(spec_loc, sp_rot)
        self.world.get_spectator().set_transform(trans)

        ##########
        # Set vehicle transform
        rotation = spawn_point.rotation
        yaw_st   = spawn_point.rotation.yaw
        x_wld, y_wld = self.local2world(x_loc, y_loc, yaw_st)
        location = spawn_point.location + carla.Location(x=x_wld, y=y_wld, z=0)
        rotation = carla.Rotation( yaw = yaw_st + psi_loc )
        trans = carla.Transform(location, rotation)
        self.vehicle.set_transform(trans)

        ##########        
        # Vehicle vlocity
        vx = self.initial_speed
        #vy = 0
        vy = 0.5* int(vx * np.random.rand(1))
        world_vx, world_vy = self.local2world(vx, vy, rotation.yaw)        
        velocity_world = carla.Vector3D(world_vx, world_vy, 0)
        self.vehicle.set_target_velocity(velocity_world) # effective after two frames

        ##########        
        # Vechicle angular velocity
        yaw_rate = np.random.uniform(-360, 360)
        angular_velocity = carla.Vector3D(z = yaw_rate)
        self.vehicle.set_target_angular_velocity(angular_velocity)

        ##########
        # Get initial state
        x_vehicle = self.getVehicleState()
        x_road    = self.getRoadState()            
        initState = x_vehicle + x_road

        return initState
                
    def generate_random_spawn_point(self):
        
        if self.custom_map_path != None: # for racing track
            #spawn_point = random_spawn_custom_map(self.all_sp)
            #print(spawn_point.location)
            # current cam spawn point carla.Transform(carla.Location(x=-1005.017395, y=203.138901, z=0.485300))
            # current car spawn point carla.Transform(carla.Location(x=-999.017395, y=203.138901, z=0.485300))
            current_spawn_point = carla.Transform(carla.Location(x=-980.518616, y=202.946793, z=0.500000))
            start_point = {'location':{'x':-1005.518188, 'y':203.016663, 'z':0.500000}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
            corner_point = {'location':{'x':-967.017395, 'y':203.016663, 'z':0.500000}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
            end_point = {'location':{'x':-967.017395, 'y':230.016663, 'z':0.500000},'rotation':{'pitch':0.000000,'yaw':89.99954,'roll':0.000000}}
            spawn_point = random_spawn_point_corner_new_map(current_spawn_point,start_point, corner_point, end_point)
            #print("-----", spawn_point)

        else: # for Town2
            world_map = self.world.get_map()
            new_spawn_point = world_map.get_spawn_points()[1]
            start_point = {'location':{'x':-7.530000, 'y':270.729980, 'z':0.500000}, 'rotation':{'pitch':0.000000,'yaw':89.99954,'roll':0.000000}}
            corner_point = {'location':{'x':-7.390556, 'y':303.114441, 'z':0.520332}, 'rotation':{'pitch':0.000000,'yaw':0.000000,'roll':0.000000}}
            end_point = {'location':{'x':25.694059, 'y':306.545349, 'z':0.521810},'rotation':{'pitch':0.000000,'yaw':0.000,'roll':0.000000}}
            spawn_point_trans = random_spawn_point_corner(new_spawn_point,start_point, corner_point, end_point)
            way_point = world_map.get_waypoint(spawn_point_trans.location, project_to_road=True)
            x_rd   = way_point.transform.location.x
            y_rd   = way_point.transform.location.y
            yaw_rd = way_point.transform.rotation.yaw
            spawn_point = carla.Transform(carla.Location(x_rd, y_rd,0.50000), 
                                          carla.Rotation(0, yaw_rd, 0))  
            
        return spawn_point

    def set_camera(self, blueprint_library, transform):
        if len(self.actor_list) == 1:
            self.IM_WIDTH, self.IM_HEIGHT = 640, 480
            cam_bp = blueprint_library.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', f'{self.IM_WIDTH}')
            cam_bp.set_attribute('image_size_y', f'{self.IM_HEIGHT}')
            cam_bp.set_attribute('fov', '110')
            camera_loc = carla.Location(transform.location.x, transform.location.y, transform.location.z + 10)
            camera_rot = carla.Rotation(pitch = -20, yaw = transform.rotation.yaw)
            spawn_point = carla.Transform(camera_loc, camera_rot)
            sensor = self.world.spawn_actor(cam_bp, spawn_point)
            self.image_queue = queue.Queue()
            sensor.listen(self.image_queue.put)
            self.actor_list.append(sensor)
        else:
            camera_loc = carla.Location(transform.location.x, transform.location.y, transform.location.z + 10)
            camera_rot = carla.Rotation(pitch = -20, yaw = transform.rotation.yaw)
            spawn_point = carla.Transform(camera_loc, camera_rot)
            self.actor_list[1].set_transform(spawn_point)
        
    ###########################################################################
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
								manual_gear_shift = True,
								gear = 4)
        
        if not self.autopilot: 
            self.vehicle.apply_control(self.control)

        # get new state
        self.world.tick()
        x_vehicle = self.getVehicleState()
        x_road    = self.getRoadState()            
        newState = x_vehicle + x_road

        # process camera image data
        if self.image_queue:
            image = self.image_queue.get()
            self.process_img(image)

        # Is done?
        location = self.vehicle.get_location()
        lat_err  = x_road[0]
        if self.world.get_map().name == 'Carla/Maps/Town02':
            if location.x > 20: # Goal
                isDone = True
                reward = 1    
            elif np.abs( lat_err ) > 1:  # Unsafe
                isDone = True
                reward = 0
            else:
                isDone  = False
                reward  = 0
        else: 
            if np.abs( lat_err ) > 8:  # Unsafe
                #print("lateral error")
                isDone = True
                reward = 0
            else:
                isDone  = False
                reward  = 0            

        return newState, reward, isDone
    
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("camera", i3)
        cv2.waitKey(1)    
        return i3/255.0    


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
        #print(f"slip -- {slip_angle} | yaw -- {yaw_rate}")
                
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
            #draw_path(wp_transform.location, self.world, life_time=1.0)
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
        if way_point.lane_width < 20:
            right_way_point = way_point.get_right_lane()
            left_way_point = way_point.get_left_lane()
            way_point = right_way_point if right_way_point.lane_width > left_way_point.lane_width else left_way_point
                
        #assert way_point.lane_width > 20, "waypoint from wrong lane"
        
        #draw_path(way_point.transform.location, self.world, life_time=10.0)
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
        
        # Destroy camera view
        if self.image_queue:
            cv2.destroyAllWindows()


##############################################################################
# main
##############################################################################
if __name__ == '__main__': 
   
    carla_port = 3429
    time_step  = 0.01
    
    try:
        rl_env = CarEnv(port= carla_port, 
                        time_step=time_step, 
                        custom_map_path="/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/train.xodr") #, autopilot=True)
        #spectate(rl_env.world)
        #rl_env.display_spawn_points()
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