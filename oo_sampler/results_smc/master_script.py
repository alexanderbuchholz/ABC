#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
from functions_smc import gaussian_kernel

#N_particles_list = [250, 500, 750, 1000, 1500, 2000]
#N_particles_list = [100, 250]
N_particles_list = [250, 500, 750, 1000]
#N_runs = 40
N_runs = 10
#epsilon = np.array([2.,1.5,1])#,0.5,0.2,0.1,0.07,0.05,0.04,0.03,0.02,0.01])
#epsilon = np.array([2.,1.5,1,0.5,0.2,0.1,0.07,0.05,0.04,0.03,0.02,0.015,0.01,0.009,0.008,0.007,0.006,0.005, 0.0045, 0.004, 0.003, 0.0025, 0.002])
epsilon = np.array([1,0.5, 0.2])
save = True
N_processes = 4
multiple_y = 10
kernel_name = "_gaussian_"
kernel = gaussian_kernel
hilbert=True
