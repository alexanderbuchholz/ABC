# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:31:59 2016
     functions for the tuberculosis model
@author: alex
"""
#import cProfile
import ipdb as pdb

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
path = '/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model/'
with open(path+'R_simulator/toggle_switch_simulator.R', 'r') as f:
    string = f.read()
toggle_switch_simulator = STAP(string, "toggle_switch_simulator")

def simulator(theta):
    """
    simulator that is based on the r code by Pierre Jacob
    """
    if theta.shape[0] != 7:
        raise ValueError('dimension of theta is not correct!')
    theta_R = robjects.FloatVector(theta)
    N = 500
    y = np.array(toggle_switch_simulator.robservation(N, theta_R))
    return y


if __name__ == '__main__':
    theta = np.array([22, 12, 4, 4.5, 325, 0.25, 0.15])
    print simulator(theta)
    #pdb.set_trace()