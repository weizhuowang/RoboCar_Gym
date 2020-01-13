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
        self.initRendering()

        #define limiting space
        self.max_bank_angle = 0.5
        high = np.array([50,50,50,50,50,50,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle,self.max_bank_angle])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([30,-200,-200,-200]),high=np.array([50,200,200,200]),dtype=np.float32)

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
        if max(np.absolute(self.state[[0,2,4]])) > 100:
            print(self.state[[0,2,4]])
            done = True
        return self.state,done


    def reset(self):
        # self.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
        self.state = np.array([-5+10*random(),0,-5+10*random(),0,0+20*random(),0,0,0,0,0,0,0],dtype=np.float32)
        if self.renderflg:
            self.drone.pos = vector(0,0,0)
            self.drone.axis = vector(1,0,0)
            self.drone.up = vector(0,1,0)
            self.xPointer.pos = self.drone.pos
            self.yPointer.pos = self.drone.pos
            self.zPointer.pos = self.drone.pos

            self.yPointer.axis = 7*self.drone.axis
            self.zPointer.axis = 7*self.drone.up
            xaxis = self.drone.axis.cross(self.zPointer.axis)
            self.xPointer.axis = xaxis

        return self.state
    
    def initRender(self):
        self.canvas = canvas(width=1200, height=900, title='Quadrotor-3D')
        ground_y = -0.5
        thk = 0.5
        ground_width = 200
        wallB = box(canvas=self.canvas,pos=vector(0, ground_y, 0), size=vector(ground_width, thk, ground_width),  color = vector(0.9,0.9,0.9))


        l_beam = self.beam_length
        r_beam = 0.4
        beam1 = cylinder(pos=vector(-l_beam/2,0,-l_beam/2),axis=vector(l_beam,0,l_beam), radius=r_beam,color=vector(0.3,0.3,0.3))
        beam2 = cylinder(pos=vector(-l_beam/2,0,l_beam/2),axis=vector(l_beam,0,-l_beam), radius=r_beam,color=vector(0.3,0.3,0.3))

        h_prop = 0.4
        r_prop = 1.5
        prop_y = 0.4
        prop1 = cylinder(pos=vector(-l_beam/2,prop_y,-l_beam/2),axis=vector(0,h_prop,0), radius=r_prop,color=color.green)
        prop2 = cylinder(pos=vector(l_beam/2,prop_y,-l_beam/2),axis=vector(0,h_prop,0), radius=r_prop,color=color.red)
        prop3 = cylinder(pos=vector(-l_beam/2,prop_y,l_beam/2),axis=vector(0,h_prop,0), radius=r_prop,color=color.purple)
        prop4 = cylinder(pos=vector(l_beam/2,prop_y,l_beam/2),axis=vector(0,h_prop,0), radius=r_prop,color=color.cyan)
        self.drone = compound([beam1, beam2, prop1, prop2, prop3, prop4],pos = vector(0,0,0),make_trail=False,retain=300)

        #yzx

        self.xPointer = arrow(pos=self.drone.pos,axis=vector(0,0,-7),shaftwidth=0.3,color=color.red)
        self.yPointer = arrow(pos=self.drone.pos,axis=self.drone.axis,shaftwidth=0.3,color=color.blue)
        self.zPointer = arrow(pos=self.drone.pos,axis=7*self.drone.up,shaftwidth=0.3,color=color.green)
        self.drone.mass = self.drone_mass
        origin = sphere(pos=vector(0,0,0), radius=0.5, color=color.yellow)

        self.x_axis = arrow(pos=vector(0,0,0),axis=vector(0,0,3),shaftwidth=0.3,color=color.red)
        self.y_axis = arrow(pos=vector(0,0,0),axis=vector(3,0,0),shaftwidth=0.3,color=color.blue)
        self.z_axis = arrow(pos=vector(0,0,0),axis=vector(0,3,0),shaftwidth=0.3,color=color.green)
        rate(FPS)

    def render(self, mode='human', close=False):
        #self.t += 0.001
        phi_   = self.state[6]
        theta_ = self.state[7]
        psi_   = self.state[8]
        #print(self.state)        
        self.drone.pos = vector(self.state[2],self.state[4],self.state[0])
        self.drone.up = vector(0,1,0)#vector(-math.sin(phi_),math.cos(phi_)+math.cos(theta_),math.sin(theta_))
        self.drone.axis = vector(1,0,0)#vector(math.cos(psi_),(math.sin(phi_)*math.cos(psi_)-math.sin(theta_)*math.sin(psi_))/(math.cos(phi_)*math.cos(theta_)),math.sin(psi_))
        self.drone.rotate(angle=phi_,axis=self.x_axis.axis)
        self.drone.rotate(angle=theta_,axis=self.y_axis.axis)
        self.drone.rotate(angle=psi_,axis=self.z_axis.axis)

        self.xPointer.pos = self.drone.pos
        self.yPointer.pos = self.drone.pos
        self.zPointer.pos = self.drone.pos

        self.yPointer.axis = 7*self.drone.axis
        self.zPointer.axis = 7*self.drone.up
        xaxis = self.drone.axis.cross(self.zPointer.axis)
        self.xPointer.axis = xaxis

        return True

    def _dsdt(self,t, s_augmented):

        # [x1,x2,r] = [x,v,psi_dot]
        x,vx,y,vy,yaw,r,torque,delta = c_augmented
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
