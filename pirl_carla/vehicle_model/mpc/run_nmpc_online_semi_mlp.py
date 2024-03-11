"""	Nonlinear MPC using MLP for only learning tire forces
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'

import time as tm
import numpy as np
import casadi
# import _pickle as pickle
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from apacrace.params import ORCA
from apacrace.models import Dynamic, Kinematic6
from apacrace.gp.utils import loadGPModel, loadGPModelVars, loadMLPModel, loadTorchModel, loadTorchModelEq, loadTorchModelImplicit
from apacrace.tracks import ETHZ
from apacrace.mpc.planner import ConstantSpeed, GetCBFStateInner, GetCBFStateOuter, get_closest_distance_fast
from apacrace.mpc.gpmpc_torch import setupNLP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import random
import os
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#####################################################################
# Tunable Params

GP_EPS_LEN = 305
mu_init = 1.
t_collect = 6.2
LR = 0.002
BETA = 0.9

def dist(a,b) :
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# Dist covered, laps completed, Lap 0 time, Lap 1 time, Lap 2 time, Lap 3 time, Lap 4 time, Mean deviation, Track boundary violation time 
statistics = []

#####################################################################
# CHANGE THIS

SAVE_RESULTS = True
ERROR_CORR = True
TRACK_CONS = False
SAVE_VIDEO = True
RUN_NO = 'without1' # From 'with_var_speeds', 'with_const_speeds' or 'without'
ITERS_EACH_STEP = 50
LOAD_MODEL = True
ACT_FN = 'relu'
SAFEGUARD = True
MAX_STEER = 0.34
VEHICLE_MODEL = 'Kinematic'
lambda_ = 10.
lambda_2 = 3.
ALPHA = 1.

torch.manual_seed(3)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

def update_friction(Df,Dr,curr_time,style='const_decay') :
	if style is 'const_decay' :
		if curr_time > 14.4 :
			Df -= Df/2200.
			Dr -= Dr/2200.
	elif style is 'sudden' :
		if curr_time > 14.3 and curr_time < 14.5:
			Df -= Df/22.
			Dr -= Dr/22.
	return Df, Dr

class DynamicModel(torch.nn.Module):
	def __init__(self, model, deltat = 0.01):
		"""
		In the constructor we instantiate four parameters and assign them as
		member parameters.
		"""
		super().__init__()
		if ACT_FN == 'relu' :
			self.act = torch.nn.ReLU()
		elif ACT_FN =='tanh' :
			self.act = torch.nn.Tanh()
		elif ACT_FN =='lrelu' :
			self.act = torch.nn.LeakyReLU()
		elif ACT_FN =='sigmoid' :
			self.act = torch.nn.Sigmoid()
		
		self.Rx = torch.nn.Sequential(torch.nn.Linear(1,1).to(torch.float64))
		
		self.Ry = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		self.Ry[0].weight.data.fill_(1.)
		# print(self.Ry[0].bias)
		self.Ry[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		# self.Ry[2].weight.data.fill_(1.)
		# print(self.Ry[0].bias)
		# print(self.Ry[0].weight)
		self.Fy = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		
		self.Fy[0].weight.data.fill_(1.)
		# print(self.Ry[0].bias)
		self.Fy[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		self.deltat = deltat
		self.model = model

	def forward(self, x, debug=False):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		# print(x.shape)
		# out = X
		deltatheta = x[:,1]
		theta = x[:,2]
		pwm = x[:,0]
		out = torch.zeros_like(x[:,3:6])
		# print(out)
		for i in range(2) :
			vx = (x[:,3] + out[:,0]).unsqueeze(1)
			vy = x[:,4] + out[:,1]
			w = x[:,5] + out[:,2]
			alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
			alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
			Ffy = self.Fy(alpha_f)[:,0]
			Fry = self.Ry(alpha_r)[:,0]
			Frx = self.Rx(vx**2)[:,0]
			Frx = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm + Frx
			
			if debug :
				print(Ffy,Fry,Frx)
			
			Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
			vx_dot = (Frx-Ffy*torch.sin(theta)+self.model.mass*vy*w)/self.model.mass
			vy_dot = (Fry+Ffy*torch.cos(theta)-self.model.mass*vx[:,0]*w)/self.model.mass
			w_dot = (Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
			out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat
		out2 = (out)
		return out2

#####################################################################
# default settings
def get_optimal_control_outer(pwm_ref,steer_ref,v,theta,x,curvature,model,nx=60,EPSILON=1e-2) :
	# theta = -theta
	# print(steer_ref,v,theta,x,curvature)
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], nx))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = x-EPSILON
		hd_left = v*math.sin(theta)
		hdd_left = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = np.linspace(model['min_pwm'], model['max_pwm'], nx)[ind[0]]
	min_steer = np.linspace(model['min_steer'], model['max_steer'], nx)[ind[1]]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control_inner(pwm_ref,steer_ref,v,theta,x,curvature,\
			      theta_,x_,curvature_,model,nx=60,ny=120,EPSILON=0.) :
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	print(curvature,curvature_)
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	elif VEHICLE_MODEL=='Dynamic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_**2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control(pwm_ref,steer_ref,state_left,curvature_left,\
			      state_right,curvature_right,model,nx=60,ny=120,EPSILON=0.05) :
	print(state_right)
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	
	# print(curvature,curvature_)
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	
	vx = state_right[2]
	vy = state_right[3]
	omega = state_right[4]
	
	alpha_f = steer - np.arctan2(omega*model['lf']+vy,vx)
	alpha_r = np.arctan2(omega*model['lr']-vy,vx)
	
	Ffy = model['Df']*np.sin(model['Cf']*np.arctan(model['Bf']*alpha_f))
	Fry = model['Dr']*np.sin(model['Cr']*np.arctan(model['Br']*alpha_r))
	v = np.sqrt(vx**2+vy**2)
	Frx = ((model['Cm1']-model['Cm2']*vx)*pwm)
	ax = (Frx-Ffy*np.sin(steer)+model['mass']*vy*omega)/model['mass']
	ay = (Fry+Ffy*np.cos(steer)-model['mass']*vx*omega)/model['mass']
	w_dot = (Ffy*model['lf']*np.cos(steer)-Fry*model['lr'])/model['Iz']
	
	# Ffy_ = model['Df']*np.sin(model['Cf']*np.arctan(model['Bf']*0.1))
	
	# ay_ = (Fry+Ffy_*np.cos(steer_)-model['mass']*vx*omega)/model['mass']
	
	# For right boundary
	d = state_right[0]
	theta = state_right[1]
	
	print("Curv : ", curvature_right)
	curvature_right = np.clip(curvature_right,-2,0.5)
	print("Theta : ", theta)
	h_right = d-EPSILON
	hd_right = vx*np.sin(theta) + vy*np.cos(theta)
	print("hd : ",hd_right+lambda_2*h_right)
	print((omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta)))
	hdd_right = (ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta))
	
	# For left boundary
	d = state_left[0]
	theta = state_left[1]
	# print("Curv : ", curvature_right)
	curvature_left = np.clip(curvature_left,-0.5,10)
	# print("Theta : ", theta)
	h_left = -d-EPSILON
	hd_left = -(vx*np.sin(theta) + vy*np.cos(theta))
	# print("hd : ",hd_right+lambda_2*h_right)
	# print((omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta)))
	hdd_left = -((ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_left)*(vx*np.cos(theta)-vy*np.sin(theta)))
	
	# h_left = -x-EPSILON
	# hd_left = -v*math.sin(theta)
	# hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
	
	cost = 100*(steer-steer_ref)**2 + (pwm-pwm_ref)**2
	cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)<0)*\
		(hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)**2
	cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)<0)*\
		(hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer


RUN_FOLDER = 'RUN_ONLINE_' + str(RUN_NO) + '_' + str(GP_EPS_LEN) + '/'
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

N_collect = int(t_collect/SAMPLING_TIME)
if not TRACK_CONS:
	SUFFIX = 'NOCONS-'+RUN_NO
else:
	SUFFIX = RUN_NO

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)
model_kin = Kinematic6(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
SIM_TIME = 20.

#####################################################################
# load mlp models

MODEL_PATH = '../gp/orca/semi_mlp-v2.pickle'
model_ = DynamicModel(model)
if LOAD_MODEL :
	model_.load_state_dict(torch.load(MODEL_PATH))


model_Rx = loadTorchModelImplicit('Rx',model_.Rx)
model_Ry = loadTorchModelImplicit('Ry',model_.Ry)
model_Fy = loadTorchModelImplicit('Fy',model_.Fy)

models = {
	'Rx' : model_Rx,
	'Ry' : model_Ry,
	'Fy' : model_Fy,
	'act_fn' : ACT_FN
}
x_train = np.zeros((GP_EPS_LEN,2+3+1))

optimizer = torch.optim.SGD(model_.parameters(), lr=LR,momentum=BETA)
loss_fn = torch.nn.MSELoss()

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, models, track, GP_EPS_LEN=GP_EPS_LEN, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR)

#####################################################################
# define load_data

def load_data(data_dyn, data_kin, VARIDX):
	y_all = (data_dyn['states'][:6,1:]-data_dyn['states'][:6,:-1]) #- data_kin['states'][:6,1:N_SAMPLES+1]
	x = np.concatenate([
		data_kin['inputs'][:,:-1].T,
		data_dyn['inputs'][1,:-1].reshape(1,-1).T,
		data_dyn['states'][3:6,:-1].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	return x, y

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
inputs_kin = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

data_dyn = {}
data_kin = {}
data_dyn['time'] = time
data_kin['time'] = time

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:n_states,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
states_kin = np.zeros([7,n_steps+1])
states_kin[:,0] = states[:,0]

# dynamic plot
H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])


# plt.figure()
# plt.grid(True)
# ax2 = plt.gca()
# LnFfy, = ax2.plot(0, 0, label='Ffy')
# LnFrx, = ax2.plot(0, 0, label='Frx')
# LnFry, = ax2.plot(0, 0, label='Fry')
# LnSpeeds, = ax2.plot(0, 0, label='Speeds')
# plt.xlim([0, SIM_TIME])
# plt.ylim([-params['mass']*9.81, params['mass']*9.81])
# plt.xlabel('time [s]')
# plt.ylabel('force [N]')
# plt.legend()
# plt.ion()

# plt.figure()
# plt.grid(True)
# ax2 = plt.gca()
# LnFry_pred, = ax2.plot(0, 0, label='Fry pred')
# LnFry_gt, = ax2.plot(0, 0, label='Fry gt')
# LnFfy_pred, = ax2.plot(0, 0, label='Ffy pred')
# LnFfy_gt, = ax2.plot(0, 0, label='Ffy gt')
# plt.xlim([-0.6, 0.6])
# plt.ylim([-2*params['Dr'], 2*params['Dr']])
# plt.xlabel('alpha [rad]')
# plt.ylabel('force [N]')
# plt.legend()
# plt.ion()

fig_mus = plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnDf, = ax2.plot(0, 0, label='max force(f)')
LnDf_pred, = ax2.plot(0, 0, label='max force predicted(f)')
LnDr, = ax2.plot(0, 0, label='max force(r)')
LnDr_pred, = ax2.plot(0, 0, label='max force predicted(r)')
plt.xlim([0, SIM_TIME])
plt.ylim([0, params['Df']*1.5])
plt.xlabel('time [s]')
plt.ylabel('lateral force [N]')
plt.legend()
plt.ion()

fig_hs = plt.figure()
plt.grid(True)
ax2 = plt.gca()
Lnh_inner, = ax2.plot(0, 0, label='inner boundary h')
Lnh_outer, = ax2.plot(0, 0, label='outer boundary h')
plt.xlim([0, SIM_TIME])
plt.ylim([-1., 1.])
plt.xlabel('time [s]')
plt.ylabel('h')
plt.legend()


fig_cbf_map = plt.figure()
plt.grid(True)
ax2 = plt.gca()
scatter_ = ax2.scatter([],[])
scatter_gt = ax2.scatter([],[])
scatter_bef = ax2.scatter([],[])
plt.xlim([-.2, 1.1])
plt.ylim([-0.37, 0.37])
plt.xlabel('pwm')
plt.ylabel('steering')
plt.legend()


# plt.figure()
# plt.grid(True)
# ax2 = plt.gca()

# plt.xlim([0, SIM_TIME])
# plt.ylim([0, params['Dr']*2.])
# plt.xlabel('time [s]')
# plt.ylabel('rear lateral force [N]')
# plt.legend()
# plt.ion()

fig_speeds = plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnSpeeds, = ax2.plot(0, 0, label='Speeds')
LnRefSpeeds, = ax2.plot(0, 0, label='Ref Speeds')
plt.xlim([0, SIM_TIME])
plt.ylim([0, 6.])
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.legend()

fig_track = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', label='Trajectory',alpha=0.8)
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=.5, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'purple', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.show()

if not os.path.exists(RUN_FOLDER):
	os.makedirs(RUN_FOLDER)
if not os.path.exists(RUN_FOLDER+'Video/'):
	os.makedirs(RUN_FOLDER+'Video/')
if not os.path.exists(RUN_FOLDER+'Video_speeds/'):
	os.makedirs(RUN_FOLDER+'Video_speeds/')
if not os.path.exists(RUN_FOLDER+'Video_mus/'):
	os.makedirs(RUN_FOLDER+'Video_mus/')
if not os.path.exists(RUN_FOLDER+'Video_hs/'):
	os.makedirs(RUN_FOLDER+'Video_hs/')
if not os.path.exists(RUN_FOLDER+'Video_cbf_map/'):
	os.makedirs(RUN_FOLDER+'Video_cbf_map/')

# main simulation loop
ref_speeds = []
Drs = []
Dfs = []
Drs_pred = []
Dfs_pred = []
Df_init = model.Df
Dr_init = model.Dr
h_outers = []
h_inners = []

dist_covered = 0
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]
cum_dists = [0.]

curr_cum_dist = 0.
for i in range(len(track.center_line[0])-1) :
	cum_dists.append(curr_cum_dist)
	curr_cum_dist += dist(track.center_line[:,i],track.center_line[:,i+1])

ref_errs = []
boundary_viol_time = 0.
v_factor = 0.9
for idt in range(n_steps-horizon):
	start_g = tm.time()
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	use_kinematic = True
	Drs.append(model.Dr)
	Dfs.append(model.Df)
	
	# load new experience into data_dyn and data_kin
	if idt > 0 : 
		start = tm.time()	
		min_ind = max(idt-GP_EPS_LEN-1,3)
		data_dyn['states'] = states[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['dstates'] = dstates[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['inputs'] = inputs[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		
		data_kin['states'] = states_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_kin['inputs'] = inputs_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
	
	
		end = tm.time()
		print("GP init time : ", end-start)
		
		y_trains = []
		for VARIDX in [3,4,5] :
			x_train, y_train = load_data(data_dyn,data_kin,VARIDX)
			y_trains.append(torch.tensor(y_train))
		y_train = torch.cat(y_trains,axis=1)
		x_train = torch.tensor(x_train)
	# Fine-tune the model
	if idt > 12 and 'without' not in RUN_NO:
		start = tm.time()	
		for i in range(ITERS_EACH_STEP) :
			# Zero your gradients for every batch!
			optimizer.zero_grad()
			outputs = model_(x_train[10:])
			loss = loss_fn(outputs, y_train[10:])
			loss.backward()
			# Adjust learning weights
			optimizer.step()
		end = tm.time()	
		print("Iter " + str(idt) + " loss : ", loss.item(), "time : ", end-start)

	model.Df, model.Dr = update_friction(model.Df, model.Dr, idt*Ts)
	params['Df'], params['Dr'] = model.Df, model.Dr
	# if idt > 720 :
	# 	model.Df -= model.Df/2200.
	# 	model.Dr -= model.Dr/2200.
	# 	params['Dr'] -= params['Dr']/2200.
	# 	params['Df'] -= params['Df']/2200.
	
	if idt > N_collect and 'without' not in RUN_NO:
		use_kinematic = False
		
	
	# planner based on BayesOpt
	prev_projidx = projidx
	if idt > N_collect and 'without' in RUN_NO:
		v_factor = 1.2
	if idt > N_collect and 'const_speeds' not in RUN_NO and 'without' not in RUN_NO:
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=(Dfs_pred[idt-1]+Drs_pred[idt-1])/(9.81*params['mass']),scale=v_factor)
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=mu_init,scale=v_factor)
	ref_speeds.append(v)
	ref_errs.append(get_closest_distance_fast(x0[:2],track))

	if projidx > 656 :
		if laps_completed > 0 :
			lap_times[laps_completed] = idt*Ts #- lap_times[laps_completed-1]
		else :
			lap_times[laps_completed] = idt*Ts
		laps_completed += 1
		projidx = 0
	else :
		dist_covered += cum_dists[projidx] - cum_dists[prev_projidx]
	projidx_inner = 0
	projidx_outer = 0
	# print(projidx)
	# solve NLP
	start = tm.time()
	projidx_inner, x_inner, theta_inner, curv_inner = GetCBFStateInner(x0=x0, track=track, projidx=projidx_inner)
	projidx_outer, x_outer, theta_outer, curv_outer = GetCBFStateOuter(x0=x0, track=track, projidx=projidx_outer)
	h_outers.append(x_outer)
	h_inners.append(-x_inner)
	if x_outer < 0 or x_inner>0 :
		print("Boundary violated")
		boundary_viol_time += Ts
	umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, use_kinematic=use_kinematic,models=model_)
	end = tm.time()
	inputs[:,idt] = np.array([umpc[0,0], states[n_states,idt] + Ts*umpc[1,0]])
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))
	control_before = inputs[:,idt].copy()
	# steer_opt, throttle_opt, map_opt, x_opt, y_opt = \
	# 	get_optimal_control_inner(inputs[0,idt],inputs[1,idt],states[3,idt],\
	# 	theta_inner,x_inner,curv_inner,theta_outer,x_outer,curv_outer,params)
	if states[1,idt] > 1. :
		curv_inner = 100.
	steer_opt, throttle_opt, map_opt, x_opt, y_opt = \
		get_optimal_control(inputs[0,idt],inputs[1,idt],[x_inner,theta_inner,states[3,idt],states[4,idt],states[5,idt]],curv_inner,\
			      [x_outer,theta_outer,states[3,idt],states[4,idt],states[5,idt]],curv_outer,params) 
	# if states[1,idt] > -0.5 or states[0,idt] > 1. :
	# 	inputs[:,idt] = np.array([throttle_opt, steer_opt])
	# update current position with numerical integration (exact model)
	x_next, dxdt_next = model.sim_continuous(states[:n_states,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	inputs_kin[:,idt] = inputs[:,idt]
	if (idt!=0) :
		inputs_kin[1,idt] = (inputs[1,idt] - inputs[1,idt-1])/Ts
	else :
		inputs_kin[1,idt] = (inputs[1,idt] - 0.)/Ts
	x_next_kin, dxdt_next_kin = model_kin.sim_continuous(states[:,idt], inputs_kin[:,idt].reshape(-1,1), [0, Ts])
	states_kin[:,idt+1] = x_next_kin[:,-1]
	
	states[:n_states,idt+1] = x_next[:,-1]
	states[n_states,idt+1] = inputs[1,idt]
	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])

	# forward sim to predict over the horizon
	steer = states[n_states,idt]
	hstates[:,0] = x0[:n_states]
	hstates2[:,0] = x0[:n_states]
	for idh in range(horizon):
		steer = steer + Ts*umpc[1,idh]
		hinput = np.array([umpc[0,idh], steer])
		x_next, dxdt_next = model.sim_continuous(hstates[:n_states,idh], hinput.reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:n_states,-1]
		hstates2[:,idh+1] = xmpc[:n_states,idh+1]

	

	# update plot
	start = tm.time()
	
	colors = []
	for m in map_opt.flatten() :
		if m :
			colors.append((0.,1.,0.))
		else :
			colors.append((1.,0.,0.))
			
	scatter_.set_offsets(np.c_[x_opt.flatten(),y_opt.flatten()])
	scatter_gt.set_offsets(np.c_[throttle_opt,steer_opt])
	scatter_bef.set_offsets(np.c_[control_before[0],control_before[1]])
	scatter_.set_color(colors)
	# scatter_.set_ydata(y_opt.flatten())
	
	LnS.set_xdata(states[0,:idt+1])
	LnS.set_ydata(states[1,:idt+1])
	
	
	# LnR.set_xdata(xref[0,1:])
	# LnR.set_ydata(xref[1,1:])

	LnP.set_xdata(states[0,idt] + dims[:,0]*np.cos(states[2,idt]) - dims[:,1]*np.sin(states[2,idt]))
	LnP.set_ydata(states[1,idt] + dims[:,0]*np.sin(states[2,idt]) + dims[:,1]*np.cos(states[2,idt]))
	
	LnH.set_xdata(hstates[0])
	LnH.set_ydata(hstates[1])

	LnH2.set_xdata(hstates2[0])
	LnH2.set_ydata(hstates2[1])
	
	alpha_f = torch.tensor(np.arange(-.6,.6,0.01)).unsqueeze(1)
	Ffy_pred = model_.Fy(alpha_f)[:,0].detach().numpy()
	Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))
	if idt < N_collect :
		Dfs_pred.append(mu_init*params['mass']*9.8*params['lr']/(params['lf']+params['lr']))
	else :
		Dfs_pred.append(np.max(Ffy_pred))
	LnDf.set_xdata(time[:idt+1])
	LnDf.set_ydata(Dfs[:idt+1])
	LnDf_pred.set_xdata(time[:idt+1])
	LnDf_pred.set_ydata(Dfs_pred[:idt+1])
	
	alpha_r = torch.tensor(np.arange(-.6,.6,0.01)).unsqueeze(1)
	Fry_pred = model_.Ry(alpha_r)[:,0].detach().numpy()
	Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))
	if idt < N_collect :
		Drs_pred.append(mu_init*params['mass']*9.8*params['lf']/(params['lf']+params['lr']))
	else :
		Drs_pred.append(np.max(Fry_pred))
	LnDr.set_xdata(time[:idt+1])
	LnDr.set_ydata(Drs[:idt+1])
	LnDr_pred.set_xdata(time[:idt+1])
	LnDr_pred.set_ydata(Drs_pred[:idt+1])
	# print((time[:idt+1]),(states[3,:idt+1]))
	
	LnSpeeds.set_xdata(time[:idt+1])
	LnSpeeds.set_ydata(states[3,:idt+1])

	Lnh_outer.set_xdata(time[:idt+1])
	Lnh_outer.set_ydata(h_outers)
	Lnh_inner.set_xdata(time[:idt+1])
	Lnh_inner.set_ydata(h_inners)
	# print((time[:idt+1]),(ref_speeds))
	LnRefSpeeds.set_xdata(time[:idt+1])
	LnRefSpeeds.set_ydata(ref_speeds)
	
	if idt%1 == 0 :
		plt.pause(Ts/10000)
	end = tm.time()	
	print("Remaining time : ", (end-start))
	if SAVE_VIDEO :
		fig_track.savefig(RUN_FOLDER+'Video/frame'+str(idt)+'.png', dpi=200)
		# fig_speeds.savefig(RUN_FOLDER+'Video_speeds/frame'+str(idt)+'.png', dpi=200)
		# fig_mus.savefig(RUN_FOLDER+'Video_mus/frame'+str(idt)+'.png', dpi=200)
		fig_hs.savefig(RUN_FOLDER+'Video_hs/frame'+str(idt)+'.png')
		fig_cbf_map.savefig(RUN_FOLDER+'Video_cbf_map/frame'+str(idt)+'.png')
	end_g = tm.time()
	print("Total time : ", (end_g-start_g))

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
	np.savez(
		'../data/DYN-GPMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME),
		time=time,
		states=states,
		dstates=dstates,
		inputs=inputs,
		)

#####################################################################
# plots

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], label='abs')
plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.plot(time[:n_steps-horizon], ref_speeds, label='ref_speeds')
plt.xlabel('time [s]')
plt.ylabel('speed [m/s]')
plt.grid(True)
plt.legend()
plt.savefig(RUN_FOLDER+'speeds.png')

# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'pwms.png')

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('steering [rad]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'steerings.png')

# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
plt.xlabel('time [s]')
plt.ylabel('orientation [rad]')
plt.grid(True)
plt.savefig(RUN_FOLDER+'orientations.png')

# plot speed
plt.figure()
plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon], label='Df')
plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon], label='Dr')
plt.xlabel('time [s]')
plt.ylabel('mu*N [N]')
plt.grid(True)
plt.legend()
plt.savefig(RUN_FOLDER+'max_friction_forces.png')
# fig_cbf_map.savefig(RUN_FOLDER+'cbf_map.png')
# fig_hs.savefig(RUN_FOLDER+'hs.png')


fig_mus.savefig(RUN_FOLDER+'mus.png')
fig_speeds.savefig(RUN_FOLDER+'speeds_online.png')
fig_track.savefig(RUN_FOLDER+'track_run.png')
# Dist covered, laps completed, Lap 0 time, Lap 1 time, Lap 2 time, Lap 3 time, Lap 4 time, Mean deviation, Track boundary violation time 
for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]

statistics.append([dist_covered,laps_completed,lap_times[0],lap_times[1],lap_times[2],lap_times[3],lap_times[4],np.mean(np.array(ref_errs)),boundary_viol_time])
np.savetxt(RUN_FOLDER+'stats.txt',np.array(statistics).T)
plt.show()

