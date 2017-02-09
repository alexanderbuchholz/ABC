# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:26:07 2017

@author: alex
Simulation started on friday 15:45 20.1.2017
"""
import numpy as np
import sys
import ipdb as pdb


sys.path.append("/home/alex/python_programming/ABC/oo_sampler/class_method_smc")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
import gaussian_densities_etc
#import functions_tuberculosis_model as functions_model
import functions_mixture_model as functions_model


Time = 40
repetitions = 20
dim_particles = 2
target_ESS_ratio_resampler = 0.4
target_ESS_ratio_reweighter = 0.4
epsilon_target = 0.1 #0.25
epsilon_start = 4
kwargs = {'N_particles_list': [500,750,1000, 1500, 2000, 2500],#, 3000, 4000, 5000], #[100,200,300,400,500,750,1000], #[1500, 2000, 2500, 3000, 4000, 5000],
            'model_description' : functions_model.model_string,
            'dim_particles' : dim_particles,
            'Time' : Time,
            'dim_auxiliary_var' : 10,
            'augment_M' : True,
            'target_ESS_ratio_reweighter' : target_ESS_ratio_resampler,
            'target_ESS_ratio_resampler' : target_ESS_ratio_reweighter,
            'epsilon_target' : epsilon_target,
            'contracting_AIS' : True,
            'M_increase_until_acceptance' : False,
            'M_target_multiple_N' : 1,
            'covar_factor' : 1.5,
            'propagation_mechanism' : 'AIS',
            'sampler_type' : 'RQMC',
            'ancestor_sampling' : False, #"Hilbert",
            'resample' : True, #True,
            'autochoose_eps' : 'ess_based',
            'save':True,
            'mixture_components' : 1,
            'y_star' : functions_model.f_y_star(dim_particles),
            'epsilon': np.linspace(epsilon_start, epsilon_target, Time),
            'kernel' : gaussian_densities_etc.gaussian_kernel,
            'move_particle' : gaussian_densities_etc.gaussian_move,
            'inititation_particles' : functions_model.theta_sampler_rqmc,
            'simulator' : functions_model.simulator,
            'delta' : functions_model.delta,
            'exclude_theta' : functions_model.exclude_theta,
            }

K_repetitions = range(repetitions)
#filename = functions_model.model_string+'_dim_'+str(dim_particles)+'_adaptive_M_autochoose_eps_gaussian_kernel'
filename = functions_model.model_string+'_adaptive_M_autochoose_eps_gaussian_kernel_1_VB_component'
if __name__ == '__main__':
    import parallel_simulation
    from functools import partial

    path = "/home/alex/python_programming/ABC_results_storage/simulation_results"
    import os
    os.chdir(path)
    filenames_list = [filename+str(k) for k in K_repetitions]

    if True: 
    # simulation RQMC
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        for i_simulation in filenames_list:
            partial_parallel_smc(i_simulation)

    if True: 
        # Simulation MC
        kwargs['inititation_particles'] = functions_model.theta_sampler_mc
        kwargs['sampler_type'] = 'MC'

        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        for i_simulation in filenames_list:
            partial_parallel_smc(i_simulation)


    if True: 
        # simulation Del Moral
        kwargs['inititation_particles'] = functions_model.theta_sampler_mc
        kwargs['sampler_type'] = 'MC'
        kwargs['kernel'] = gaussian_densities_etc.uniform_kernel
        kwargs['propagation_mechanism'] = 'Del_Moral'
        kwargs['M_increase_until_acceptance'] = False
        kwargs['augment_M'] = False
        kwargs['covar_factor'] = 2

        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        for i_simulation in filenames_list:
            partial_parallel_smc(i_simulation)

    if False: 
        # simulation Sisson
        kwargs['propagation_mechanism'] = 'true_sisson'
        kwargs['autochoose_eps'] = ''
        kwargs['dim_auxiliary_var'] = 1
        #kwargs['epsilon'] =  np.linspace(epsilon_start, epsilon_target, Time),
        
        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        for i_simulation in filenames_list:
            partial_parallel_smc(i_simulation)