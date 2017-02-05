# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:26:07 2017

@author: alex
Simulation started on tuesday 11:30 16.1.2017
"""
import numpy as np
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/class_method_smc")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
import gaussian_densities_etc
import functions_mixture_model
Time = 30
repetitions = 20
dim_particles = 2
kwargs = {'N_particles_list': [500,1000,1500,2000,3000,4000,5000],
            'dim_particles' : dim_particles,
            'Time' : Time,
            'dim_auxiliary_var' : 2,
            'augment_M' : True,
            'target_ESS_ratio' : 0.3,
            'epsilon_target' : 0.05,
            'contracting_AIS' : False,
            'covar_factor' : 2,
            'propagation_mechanism' : 'AIS',
            'sampler_type' : 'RQMC',
            'ancestor_sampling' : False, #"Hilbert",
            'resample' : False, #True,
            'autochoose_eps' : True,
            'save':True,
            'mixture_components' : 1,
            'y_star' : np.zeros(dim_particles),
            'epsilon': np.arange(Time,0,-1)/20.,
            'kernel' : gaussian_densities_etc.uniform_kernel,
            'move_theta' : gaussian_densities_etc.student_move,
            'inititation_particles' : functions_mixture_model.theta_sampler_rqmc,
            'simulator' : functions_mixture_model.simulator,
            'delta' : functions_mixture_model.delta,
            'exclude_theta' : functions_mixture_model.exclude_theta,
            'modified_sampling': ""}

K_repetitions = range(repetitions)
filename = 'adaptive_M_autochoose_eps_uniform_kernel'

if __name__ == '__main__':
    import parallel_simulation
    from functools import partial

    path = "/home/alex/python_programming/ABC_results_storage/simulation_results"
    import os
    os.chdir(path)
    filenames_list = [filename+str(k) for k in K_repetitions]
    partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
    #partial_parallel_smc(filenames_list[0])
    #partial_parallel_smc(filenames_list[1])
    parallel_simulation.parmap(partial_parallel_smc,filenames_list)
