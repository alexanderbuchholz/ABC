#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import pickle
import sys
sys.path.append("/home/abuchholz/python_programming/ABC/oo_sampler")
sys.path.append("/home/abuchholz/python_programming/ABC/oo_sampler/functions")
sys.path.append( "/home/abuchholz/python_programming/ABC/oo_sampler/results_smc")
sys.path.append( "/home/abuchholz/python_programming/ABC/oo_sampler/functions/mixture_model")
from functions_smc import *
from master_script import *
from functions_mixture_model import *

# specify elements of algorithm here
theta_sampler = theta_sampler_rqmc
random_sequence = random_sequence_rqmc
name = "mixture_model_multiple_y_DICT_IS_large_rqmc_run"+kernel_name
IS = False 
if True:
	from multiprocessing import Pool
	from functools import partial
	for N_particles in N_particles_list:
		name = name+str(N_particles)+"_"
		def parallel_smc(name, theta_sampler, move_theta, save, smc_abc__):
			smc_abc__.initialize_sampler(theta_sampler)
			smc_abc__.loop_over_time(move_theta, save = save, name=name)
		# construct class
		smc_abc__ = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence, kernel, exclude_theta, 2, IS=IS, multiple_y = multiple_y, hilbert=hilbert)
		#partialize
		partial_parallel_smc = partial(parallel_smc, theta_sampler = theta_sampler, move_theta = move_theta, save = save, smc_abc__ = smc_abc__)	
		list_runs = [name+str(i) for i in range(N_runs)]
	
		# start parallel computation
		pool = Pool(processes=N_processes)

		results = pool.map(partial_parallel_smc, list_runs)



