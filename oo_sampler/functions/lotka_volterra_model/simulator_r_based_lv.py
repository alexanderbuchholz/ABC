# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:31:59 2016
     functions for the lotka volterra model
@author: alex
"""
#import cProfile
import ipdb as pdb

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
path = '/home/alex/python_programming/ABC/oo_sampler/functions/lotka_volterra_model/'
with open(path+'R_simulator/r_simulator.R', 'r') as f:
    string = f.read()
r_simulator = STAP(string, "r_simulator")

def simulator(theta):
    """
    simulator that is based on the r code by Pierre Jacob
    """
    if theta.shape[0] != 3:
        raise ValueError('dimension of theta is not correct!')
    theta_R = robjects.FloatVector(theta)
    y = np.array(r_simulator.robservation(theta_R))
    return y

def y_star_function():
    y = np.array(r_simulator.y_star())
    return y


if __name__ == '__main__':
    theta = np.array([1, 0.005, 0.6])
    print simulator(theta)
    print y_star_function()
    pdb.set_trace()