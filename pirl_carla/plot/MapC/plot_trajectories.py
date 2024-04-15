import os
import numpy as np
import matplotlib.pyplot as plt


# Plot trajectories
log_dir = 'data_trained/'
files   = os.listdir(log_dir)
files.sort()

for f in files:

    data = np.load(log_dir+f)    

    x = data['position'][1,:] # * 3 - 25
    y = data['position'][0,:] # + 1080
    plt.plot(x, y, color='blue', lw=0.5, zorder=10)
    plt.scatter(x[0], y[0], color='blue', marker='x')


# Road
road_info = np.load('spawn_points.npz')
c_line    = road_info['center']
x =  c_line[:,1]
y =  c_line[:,0]
plt.plot(x, y, 'k--', zorder=-1)

boundary  = road_info['left']
x =  boundary[:,1]
y =  boundary[:,0]
plt.plot(x, y, 'k-')

boundary  = road_info['right']
x =  boundary[:,1]
y =  boundary[:,0]
plt.plot(x, y, 'k-')

plt.xlim([190,250])
plt.ylim([-1005,-955])

plt.xlabel('y [m]')
plt.ylabel('x [m]')

plt.savefig('trajectories_mapC.png', bbox_inches="tight")