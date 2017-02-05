# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:44:58 2016

@author: abuchholz
"""

from functions_smc import *
# test functions for smc

theta_old = np.squeeze(theta_sampler_mc(0,2,1)[:2,])
u = random_sequence_mc(3,0,10)
sigma = np.array([[2,1],[1,2]])
gaussian_density(theta_old, theta_old+1, sigma)
print gaussian([0.5,0.5])

print theta_old

weights = np.array([0.1,0.2,0.7])
u = 0.31
weighted_choice(weights, u)
