import gym
from gym import error, spaces, utils

from gym.utils import seeding

import numpy as np
import math
import scipy.integrate
import sympy as sp

from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow, curve, compound,label

import controlpy
from random import random

FPS = 200

class RaceCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    ##################
    # Control parameters:
    # delta (front wheel angle)
    # torque (F)
    # 
    # fxr = 0.5*F
    # fxf = 0.5*F*sin(delta)
    # fyf = 0.5*F*cos(delta)
    # 
    # Need output parameters:
    # xd,yd,yaw_d,xdd,ydd,yaw_dd
    # 
    # Need input parameters:
    # x y psi delta v a lf lr
    # 
    ##################

    def __init__(self):

        self.gravity = 9.81

        self.car = None
        self.state = np.zeros(6)
        self.input = np.zeros(4)
        self.motor_mass = 2
        self.car_mass = self.motor_mass

		# Notation:
		# lr: Dist of CG to Front Axle
		# lf: Dist of CG to Rear  Axle

        self.lf = 0.15
        self.lr = 0.15

        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        self.t = 0
        self.dt = 0.02
        self.moi = self.car_mass/12*0.29
        
        self.pointer = None
        self.viewer = None

        #define limiting space
        high = np.array([10,10,10,10,5,5])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-0.1,-0.01]),high=np.array([0.1,0.01]),dtype=np.float32)

    def setrender(self,renderflg):
        self.renderflg = renderflg
        if renderflg:
            self.initRender()

    def setdt(self,dt):
        self.dt = dt

    def step(self, action):
        self.input = action
        
        state_augmented = np.append(self.state, self.input)
        sol = scipy.integrate.solve_ivp(self._dsdt, [0, self.dt], state_augmented)

        ns = sol.y[:,-1]
        self.state = ns[:-2]

        done = False
        #if the car is going out of bounds, terminate this episode
        if max(np.absolute(self.state[[0,2]])) > 10:
            print(self.state)
            done = True
        return self.state,done

# State: [x,vx,y,vy,yaw,yaw_d] Action: [Torque, steering]

    def reset(self):
        # self.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
        self.state = np.array([-5+10*random(),0,-5+10*random(),0,-0.5+random(),0],dtype=np.float32)
        if self.renderflg:
            self.render()
        return self.state
    
    def initRender(self):
        self.canvas = canvas(width=1200, height=900, title='Car-3D')
        # self.canvas = canvas(width=1660, height=1010, title='Car-3D')
        ground_y = 0.17
        thk = 0.1
        ground_width = 10.0
        wallB = box(canvas=self.canvas,pos=vector(0, -ground_y, 0), size=vector(ground_width, thk, ground_width),  color = color.blue)

            # vector:(front/rear,z,l/r)

        car_w,car_l,car_h = 0.132, 0.26/0.6, 0.2
        wheel_r,wheel_h   = 0.095,0.035
        chassis = box(canvas=self.canvas, pos=vector(0,wheel_r+0.05,0),size=vector(car_l,car_h,car_w),color=color.green)
        

        wheel_r/2+thk
        wheel1 = cylinder(pos=vector(0.6*car_l/2.0 ,wheel_r ,car_w/2.0),axis=vector(0,0,wheel_h), radius=wheel_r,color=color.blue)
        wheel2 = cylinder(pos=vector(0.6*car_l/2.0 ,wheel_r ,-car_w/2.0-wheel_h),axis=vector(0,0,wheel_h), radius=wheel_r,color=color.blue)
        wheel3 = cylinder(pos=vector(-0.6*car_l/2.0,wheel_r ,car_w/2.0),axis=vector(0,0,wheel_h), radius=wheel_r,color=color.blue)
        wheel4 = cylinder(pos=vector(-0.6*car_l/2.0,wheel_r ,-car_w/2.0-wheel_h),axis=vector(0,0,wheel_h), radius=wheel_r,color=color.blue)
        
        self.car = compound([chassis,wheel3, wheel4],pos = vector(self.x_pos,self.y_pos,self.z_pos),make_trail=True,retain=300)
        self.frontwheel = compound([wheel1, wheel2],pos = vector(self.x_pos,self.y_pos,self.z_pos))
        self.car.axis = vector(1,0,0)
        self.car.up = vector(0,1,0)
        self.car.mass = self.car_mass

        self.pointer = arrow(pos=self.car.pos,axis=self.car.axis,shaftwidth=0.01)
        origin = sphere(pos=vector(0,0,0), radius=0.02)

        self.x_axis = arrow(pos=vector(0,0,0),axis=vector(0,0,3),shaftwidth=0.3,color=color.red)
        self.y_axis = arrow(pos=vector(0,0,0),axis=vector(3,0,0),shaftwidth=0.3,color=color.blue)
        self.z_axis = arrow(pos=vector(0,0,0),axis=vector(0,3,0),shaftwidth=0.3,color=color.green)
        rate(FPS)

    def render(self, mode='human', close=False):

        car_w,car_l,car_h = 0.132, 0.26/0.6, 0.2
        wheel_r,wheel_h   = 0.095,0.035
        self.car.pos  = vector(self.state[0],0,self.state[2])
        self.car.axis = vector(math.cos(self.state[4]),0.000001,math.sin(self.state[4]))
        self.frontwheel.pos  = vector(self.state[0],-0.03,self.state[2])+0.6*car_l/2.0*self.car.axis
        self.frontwheel.axis = vector(math.cos(self.state[4]+self.input[1]),0.000001,math.sin(self.state[4]+self.input[1]))

        self.pointer.pos  = self.car.pos
        self.pointer.axis = self.car.axis

        return True

    def _dsdt(self,t, s_augmented):

        # [x1,x2,r] = [x,v,psi_dot]
        x,vx,y,vy,yaw,r,torque,delta = s_augmented
        v = math.sqrt(vx**2 + vy**2)

        # Self made friction model
        torque = torque-0.2*v**2

        Fxr = torque/0.095
        Fxf = 0
        Fyf = 0
        Fyr = 0

        beta = math.atan(self.lr/(self.lf+self.lr)*math.tan(delta))

        xd    = vx*math.cos(yaw)-vy*math.sin(yaw)
        yd    = vx*math.sin(yaw)+vy*math.cos(yaw)
        # yawd  = r
        # Use any one, they are the same numerically
        yawd  = v/self.lr*math.sin(beta)                              # easier expression
        # yawd  = v*math.cos(beta)/(self.lf+self.lr)*math.tan(delta)    # longer expression
        xdd   = vy*r  + (1/self.car_mass)*(Fxr - Fyf*math.sin(delta))
        ydd   = -vx*r + (1/self.car_mass)*(Fyr + Fyf*math.cos(delta))
        yawdd = (1/self.moi)*(self.lf*Fyr*math.cos(delta)-self.lf*Fyr)

        return xd,xdd,yd,ydd,yawd,yawdd,0,0

    def bound(self,x,m,M):
        return min(max(x, m), M)

    def limitTorque(self,torque,axis):
        #print('torque: '+str(torque)+' on ' + axis)
        if axis == 'x':
            return min(torque,300)
        else:
            return min(torque,300)
