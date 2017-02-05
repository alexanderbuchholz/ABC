# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:39:46 2017
    simulation of tuberculosis model
@author: alex
"""

import numpy as np
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/class_method_smc")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
import gaussian_densities_etc
import functions_tuberculosis_model as functions_model
Time = 20
repetitions = 20
kwargs = {'N_particles_list': [100, 200, 500,1000,2000],
            'dim_particles' : 2,
            'Time' : Time,
            'dim_auxiliary_var' : 10,#20,
            'covar_factor' : 2,
            'propagation_mechanism' : 'AIS',
            'sampler_type' : 'RQMC',
            'ancestor_sampling' : False, #"Hilbert",
            'resample' : False, #True,
            'autochoose_eps' : True,
            'save':True,
            'mixture_components' : 10,
            'y_star' : functions_model.y_star,
            'epsilon': np.arange(Time,0,-1)/10.,
            'kernel' : gaussian_densities_etc.gaussian_kernel,
            'move_theta' : gaussian_densities_etc.move_theta,
            'inititation_particles' : functions_model.theta_sampler_rqmc,
            'simulator' : functions_model.simulator,
            'delta' : functions_model.delta,
            'exclude_theta' : functions_model.exclude_theta}

K_repetitions = range(repetitions)
filename = 'tuberculosis_model_'

if __name__ == '__main__':
    import parallel_simulation
    from functools import partial

    path = "/home/alex/python_programming/ABC_results_storage/simulation_results"
    import os
    os.chdir(path)
    filenames_list = [filename+str(k) for k in K_repetitions]
    partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)

    parallel_simulation.parmap(partial_parallel_smc,filenames_list)
