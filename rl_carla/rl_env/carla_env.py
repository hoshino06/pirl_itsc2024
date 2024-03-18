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
            """
            for sp in wp_list:
                draw_path(sp.transform.location, world, life_time= 600)
            set_spectator(world, wp_list[0].transform)
            print(start_wp.road_id, start_wp.lane_id, start_wp.section_id, start_wp.lane_width)
            exit()
            """
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

######################################
# Get spawn point for racing track
def random_spawn_custom_map(all_spawn_point_list):
    rand_1 = np.random.randint(0,len(all_spawn_point_list))
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
###############################################################################
class CarEnv:

    step_T_pool = np.linspace(0.6,1, num=5)    # throttle values
    step_S_pool = np.linspace(-0.8,0.8, num=5) # steering angle values
    action_num  = len(step_T_pool) * len(step_S_pool)

    def __init__(self, 
                 port=2000, time_step = 0.01, autopilot=False, 
                 custom_map_path = None, 
                 actor_filter  = 'vehicle.audi.tt', # or use 'model3' 
                 spawn_method  = None,
                 vehicle_reset = None,
                 camera_view   = None, 
                 initial_speed = 10):
        """
        Connect to Carla server, spawn vehicle, and initialize variables 
        """
        # Initialization of attributes
        self.actor_list  = []
        self.image_queue = None # for camera
        self.autopilot   = autopilot
        self.actor_filter          = actor_filter
        self.spawn_method          = spawn_method
        self.vehicle_reset_method  = vehicle_reset
        self.camera_view           = camera_view
        self.initial_speed         = initial_speed

        ##############################
        # Load Carla world and map
        ##############################
        # Client creation and world connection
        print(f'Connecting carla server on port {port}.')
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.custom_map_path = custom_map_path

        # load custom map and get all spawn points
        if self.custom_map_path: 
            self.world = load_custom_map(self.custom_map_path, self.client)
            self.all_sp = fetch_all_spawn_point(self.world)
            #for sp in self.all_sp:
            #    draw_path(sp.location, self.world, life_time= 600)
        else:
            if not self.world.get_map().name == 'Carla/Maps/Town02':
                self.world  = self.client.load_world('Town02')
            self.all_sp = self.world.get_map().get_spawn_points()
        
        # Set synchronous mode settings
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = time_step
        self.time_step = time_step
        self.world.apply_settings(new_settings)

        ##############################
        # Vehicle 
        ##############################
        # Create vehicle actor        
        blueprint_library = self.world.get_blueprint_library()
        self.bl = blueprint_library
        vehicle_bp = blueprint_library.filter(self.actor_filter)[0]
        if self.spawn_method:         
            spawn_point = self.spawn_method(self)
        else:
            spawn_point = self.all_sp[1]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        if self.autopilot:
            self.vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
            print('Vehicle was spawned with auto pilot mode')

        # Change paramters
        physics_control = self.vehicle.get_physics_control()
        wheels = physics_control.wheels
        wheels[0].tire_friction = 2.75
        wheels[0].lat_stiff_value = 40.
        wheels[1].tire_friction = 2.75
        wheels[1].lat_stiff_value = 40.
        wheels[2].tire_friction = 2.75
        wheels[2].lat_stiff_value = 40.
        wheels[3].tire_friction = 2.75
        wheels[3].lat_stiff_value = 40.
        steer_curve = physics_control.steering_curve
        steer_curve[0].y = 1.
        steer_curve[1].y = 1.
        steer_curve[2].y = 1.
        steer_curve[3].y = 1.
        physics_control.steering_curve = steer_curve
        physics_control.wheels = wheels
        physics_control.center_of_mass.x = 0.
        self.vehicle.apply_physics_control(physics_control)
        print("Modified physics control")
        physics_control = self.vehicle.get_physics_control()
        #print(physics_control)

        ##############################
        # Camera view 
        ##############################
        # Change spectator view (carla window)
        sp_loc   = spawn_point.location
        sp_rot   = spawn_point.rotation
        spec_loc = sp_loc + carla.Location(x=0, y=0, z=5)         
        trans    = carla.Transform(spec_loc, sp_rot)
        self.world.get_spectator().set_transform(trans)
        
        # Create camera actor (only for Town2)
        if self.world.get_map().name == 'Carla/Maps/Town02':
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

        # Set additional camera view
        if self.camera_view:
            self.set_camera(self.bl, spawn_point)

        # Run one step and store state dimension
        initState = self.reset()
        self.state_dim  = len(initState)

    ###########################################################################
    # reset
    def reset(self):
        """
        Initialize vehicle state
        """
        ##########
        # Choose spawn point and set camera
        if self.spawn_method:
            spawn_point = self.spawn_method(self)
        else:            
            spawn_point = self.generate_random_spawn_point()
        
        # Change spectator view (carla window)
        sp_loc   = spawn_point.location
        sp_rot   = spawn_point.rotation
        spec_loc = sp_loc + carla.Location(x=0, y=0, z=5)         
        trans    = carla.Transform(spec_loc, sp_rot)
        self.world.get_spectator().set_transform(trans)

        ##########
        # Vehicle position and heading angle (wrt spanw point)
        if self.vehicle_reset_method is None: 
            # position and angle
            x_loc    = 0
            y_loc    = 0
            psi_loc  = 0
            # velocity and yaw rate
            vx = self.initial_speed
            vy = 0
            yaw_rate = 0
        else:             
            x_loc, y_loc, psi_loc, vx, vy, yaw_rate = self.vehicle_reset_method()
        #print(x_loc, y_loc, psi_loc, vx, vy, yaw_rate)
            
        ##########
        # Set vehicle transform
        rotation = spawn_point.rotation
        yaw_st   = spawn_point.rotation.yaw
        x_wld, y_wld = self.local2world(x_loc, y_loc, yaw_st)
        location = spawn_point.location + carla.Location(x=x_wld, y=y_wld, z=0)
        rotation = carla.Rotation( yaw = yaw_st + psi_loc )
        trans = carla.Transform(location, rotation)
        self.vehicle.set_transform(trans)
     
        # Set vehicle vlocity
        world_vx, world_vy = self.local2world(vx, vy, rotation.yaw)        
        velocity_world = carla.Vector3D(world_vx, world_vy, 0)
        self.vehicle.set_target_velocity(velocity_world) # effective after two frames
     
        # Set Vechicle angular velocity
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
            spawn_point = random_spawn_custom_map(self.all_sp)
            #print(spawn_point.location)

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
    # step
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
								#manual_gear_shift = False,
								#gear = 0, 
                                manual_gear_shift = True,
                                gear = 4,
                                )
        
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
            if location.x > 20: # Goal (only for Town02)
                isDone = True
                reward = 1    
            elif np.abs( lat_err ) > 1:  # Unsafe
                isDone = True
                reward = 0
            else:
                isDone  = False
                reward  = 0
        else: 
            if np.abs( lat_err ) > 10:  # Unsafe
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

    ###########################################################################
    # Utils
    def get_all_spawn_points(self):
        return self.all_sp

    def get_vehicle_position(self):
        vehicle_trans = self.vehicle.get_transform()  
        vehicle_locat = vehicle_trans.location
        vehicle_rotat = vehicle_trans.rotation 
        x   = vehicle_locat.x
        y   = vehicle_locat.y
        yaw = vehicle_rotat.yaw        
        return [x,y,yaw]
    
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
# Test code for carl_env
##############################################################################
if __name__ == '__main__': 

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """        

    carla_port = 3000
    time_step  = 0.05

    # maps 
    map_simple       = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/simple.xodr"
    map_train        = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/train.xodr"
    map_test         = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test.xodr"
    map_test_refined = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test_refined.xodr"
    map_zhenhua      = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/zhenhua.xodr"
    map_town2        = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/Town02.xodr"

    # spawn method (initial vehicle location)
    def choose_spawn_point(carla_env):
        sp_list = carla_env.get_all_spawn_points()    
        spawn_point = sp_list[0]
        return spawn_point

    def random_spawn_point(carla_env):
        sp_list     = carla_env.get_all_spawn_points()       
        rand_1      = np.random.randint(0,len(sp_list))
        spawn_point = sp_list[rand_1]
        return spawn_point

    # vehicle state initialization
    def vehicle_reset_method(): 
        
        # position and angle
        x_loc    = 0
        y_loc    = 0 #np.random.uniform(-5,5)
        psi_loc  = 0
        # velocity and yaw rate
        vx = 20
        vy = 0
        yaw_rate = 0        
        
        # It must return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]

    ##########################################################################
    # simulation
    try:
        rl_env = CarEnv(port=carla_port, 
                        time_step=time_step,
                        custom_map_path=map_town2, # None: Town2
                        spawn_method=random_spawn_point, # None: random pick
                        actor_filter = 'vehicle.audi.tt',  
                        vehicle_reset= vehicle_reset_method, 
                        autopilot=True)

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
