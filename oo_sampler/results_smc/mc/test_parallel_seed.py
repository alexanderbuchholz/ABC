#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import pickle
import sys
sys.path.append("/home/aa11eexx/python_programming/ABC/oo_sampler")
sys.path.append( "/home/aa11eexx/python_programming/ABC/oo_sampler/results_smc")
from functions_smc import *
from master_script import *
from functions_file import *
import numpy as np

if True:
	from multiprocessing import Pool
	from functools import partial
	def print_random(i):
		theta = np.array([0.5,0.1,0.4])
		theta = theta.reshape((3,1))
		print simulator(theta)
#		print(i)
#		print(random_sequence_mc(1,0,1))
	list_runs = ["mc_run"+str(i) for i in range(N_runs)]
#	partial_parallelizable_smc(list_runs[0])
	pool = Pool(processes=N_processes)

	results = pool.map(print_random, list_runs)


