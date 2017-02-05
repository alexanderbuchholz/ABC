#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import pickle
import sys
sys.path.append("/home/aa11eexx/python_programming/ABC/oo_sampler")
sys.path.append("/home/aa11eexx/python_programming/ABC/oo_sampler/functions")
sys.path.append( "/home/aa11eexx/python_programming/ABC/oo_sampler/results_smc")
from functions_smc import *
from master_script import *
from functions_file import simulator

from functions_smc import gaussian_kernel
import cProfile

N_particles = 10
N_runs = 2
epsilon = np.array([2.,1.5,1])#,0.5,0.2,0.1,0.07,0.05,0.04,0.03,0.02,0.01])
#epsilon = np.array([2.,1.5,1,0.5,0.2,0.1,0.07,0.05,0.04,0.03,0.02,0.015,0.01,0.009,0.008,0.007,0.006,0.005, 0.0045, 0.004, 0.003, 0.0025])
save = False
N_processes = 2
kernel_name = "_gaussian_"
kernel = gaussian_kernel
IS = True

# specify elements of algorithm here
theta_sampler = theta_sampler_rqmc
random_sequence = random_sequence_rqmc
name = "profile_rqmc_run"+kernel_name

if True:
        from multiprocessing import Pool
        from functools import partial
        def parallel_smc(name, theta_sampler, move_theta, save, smc_abc__):
                smc_abc__.initialize_sampler(theta_sampler)
                smc_abc__.loop_over_time(move_theta, save = save, name=name)
        # construct class
        smc_abc__ = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence, kernel, 2, IS=IS)
        #partialize
        partial_parallel_smc = partial(parallel_smc, theta_sampler = theta_sampler, move_theta = move_theta, save = save, smc_abc__ = smc_abc__)
        list_runs = [name+str(i) for i in range(N_runs)]

        # start parallel computation
        pool = Pool(processes=N_processes)

        cProfile.run( 'partial_parallel_smc(list_runs[0])')

if False:
	theta = np.array([[0.4],[0.3],[0.3]])
        cProfile.run('simulator(theta)', sort=1)
	print simulator(theta)

