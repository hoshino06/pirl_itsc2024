ITSC2024data/MapC/hoshino

03210235: 10m/s, without sigmoid

03230325: 10m/s, with sigmoid

03250427: 20m/s, with sigmoid
    # position and angle
    x_loc    = 0
    y_loc    = 0 #np.random.uniform(-5,5)
    psi_loc  = np.random.uniform(-30,30)
    # velocity and yaw rate
    vx = 20
    vy = 0.5*float(vx * np.random.rand(1)) 
    yaw_rate = np.random.uniform(-360,360)       

04030520: 15-25m/s, with sigmoid; waypoint_itvl=3.0
    # position and angle
    x_loc    = 0
    y_loc    = np.random.uniform(0,3)
    psi_loc  = 0 #np.random.uniform(-20,20)
    # velocity and yaw rate
    vx = np.random.uniform(15,25)
    rand_num = np.random.uniform(-0.5, 0)
    vy       = 0.5*vx*rand_num 
    yaw_rate = -90*rand_num 

04050046: 30m/s, with sigmoid; fixed spawn point 
    # position and angle
    x_loc    = 0
    y_loc    = 0 
    psi_loc  = 0 #np.random.uniform(-20,20)
    # velocity and yaw rate
    vx       = 30 #np.random.uniform(15,25)
    rand_num = np.random.uniform(-0.6, -0.9)
    vy       = 0.5*vx*rand_num 
    yaw_rate = -80*rand_num 

04071140-19k: 30m/s, fixed spawn point, lr=1e-4
    # horizon
    horizon     = np.random.uniform(2.5, 5)
    # position and angle
    x_loc    = 0
    y_loc    = 0 
    psi_loc  = 0 #np.random.uniform(-20,20)
    # velocity and yaw rate
    vx       = 30 #np.random.uniform(15,25)
    rand_num = np.random.uniform(-0.75, -0.85)
    vy       = 0.5*vx*rand_num 
    yaw_rate = -80*rand_num 
