# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:26:07 2017

@author: alex
Simulation started on friday 15:45 20.1.2017
"""
import numpy as np
import sys
import ipdb as pdb

root_path = "/home/alex/python_programming"
#root_path = "/home/ubuntu"

sys.path.append(root_path+"/ABC/oo_sampler/class_method_smc")
sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/parallelize")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")

#path = "/media/alex/Transcend/ABC_results_storage/simulation_results_18-4-17"
path = root_path+"/ABC_results_storage/simulation_results_18-7-17"
import gaussian_densities_etc
#import functions_tuberculosis_model as functions_model
import functions_mixture_model as functions_model
#import functions_mixture_model as functions_model


Time = 600
repetitions = 49
dim_particles = 3
target_ESS_ratio_resampler = 0.5
target_ESS_ratio_reweighter = 0.5
epsilon_target = 0.25#functions_model.epsilon_target(dim_particles) #0.001 #0.25
epsilon_start = 4
kwargs = {'N_particles_list': [1000],# 500, 750, 1500,   2500, 3000, 4000, 5000],#,],#,  3000, 4000, 5000], #[100,200,300,400,500,750,1000], #[1500, 2000, 2500, 3000, 4000, 5000],
            'model_description' : functions_model.model_string,
            'dim_particles' : dim_particles,
            'Time' : Time,
            'dim_auxiliary_var' : 10,
            'augment_M' : False,
            'M_incrementer' : 2,
            'target_ESS_ratio_reweighter' : target_ESS_ratio_resampler,
            'target_ESS_ratio_resampler' : target_ESS_ratio_reweighter,
            'epsilon_target' : epsilon_target,
            'contracting_AIS' : True,
            'M_increase_until_acceptance' : False, # False
            'M_target_multiple_N' : target_ESS_ratio_reweighter,
            'covar_factor' : 1.2,
            'propagation_mechanism' : 'AIS',
            'sampler_type' : 'QMC',
            'ancestor_sampling' : False, #"Hilbert",
            'resample' : True, #True,
            'autochoose_eps' : 'ess_based',
            'save':True,
            'mixture_components' : 1,
            'y_star' : functions_model.f_y_star(dim_particles),
            'epsilon': np.linspace(epsilon_start, epsilon_target, Time),
            'kernel' : gaussian_densities_etc.gaussian_kernel,
            'move_particle' : gaussian_densities_etc.gaussian_move,
            'inititation_particles' : functions_model.theta_sampler_qmc,
            'simulator' : functions_model.simulator,
            'delta' : functions_model.delta,
            'exclude_theta' : functions_model.exclude_theta,
            'computational_budget' : 10**10,
            'parallelize' : True,
            'fixed_epsilon_schedule': True,
            'algorithm_only_schedule': True,
            'y_simulation': 'standard',
            'start_phase_ais': 5,
            'truncate_neg_binomial': True,
            'quantile_target' : 0.3
            }

K_repetitions = range(repetitions)
#filename = functions_model.model_string+'_dim_'+str(dim_particles)+'_adaptive_M_autochoose_eps_gaussian_kernel'

if __name__ == '__main__':
    import parallel_simulation
    from functools import partial
    import functions_parallelize
    import os
    os.chdir(path)
    filename_base = functions_model.model_string+'_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3'
    #filenames_list = [filename+'_'+str(k) for k in K_repetitions]
    #filenames_list = filenames_list[9:]



    if True: 
    # simulation QMC
        kwargs['propagation_mechanism'] = 'AIS'
        kwargs['sampler_type'] = 'QMC'
        kwargs['augment_M'] = False
        kwargs['inititation_particles'] = functions_model.theta_sampler_qmc
        kwargs['kernel'] = gaussian_densities_etc.uniform_kernel
        kwargs['covar_factor'] = 1.2
        kwargs['M_increase_until_acceptance'] = False
        kwargs['y_simulation'] = 'standard'
        kwargs['autochoose_eps'] = 'quantile_based'
        kwargs['dim_auxiliary_var'] = 1
        kwargs['parallelize'] = False
        kwargs['resample'] = False
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        filename = filename_base+'_m'+str(kwargs['dim_auxiliary_var'])
        filenames_list = [filename+'_'+str(k) for k in K_repetitions]
        partial_parallel_smc(filenames_list[0])
        functions_parallelize.parallelize_partial_over_chunks(partial_parallel_smc, filenames_list[1:])

    if True:
        kwargs['dim_auxiliary_var'] = 2
        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        filename = filename_base+'_m'+str(kwargs['dim_auxiliary_var'])
        filenames_list = [filename+'_'+str(k) for k in K_repetitions]
        partial_parallel_smc(filenames_list[0])
        functions_parallelize.parallelize_partial_over_chunks(partial_parallel_smc, filenames_list[1:])

    if True:
        kwargs['dim_auxiliary_var'] = 5
        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        filename = filename_base+'_m'+str(kwargs['dim_auxiliary_var'])
        filenames_list = [filename+'_'+str(k) for k in K_repetitions]
        partial_parallel_smc(filenames_list[0])
        functions_parallelize.parallelize_partial_over_chunks(partial_parallel_smc, filenames_list[1:])

    if True:
        kwargs['dim_auxiliary_var'] = 10
        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        filename = filename_base+'_m'+str(kwargs['dim_auxiliary_var'])
        filenames_list = [filename+'_'+str(k) for k in K_repetitions]
        partial_parallel_smc(filenames_list[0])
        functions_parallelize.parallelize_partial_over_chunks(partial_parallel_smc, filenames_list[1:])

    if True:
        kwargs['dim_auxiliary_var'] = 20
        del partial_parallel_smc
        partial_parallel_smc = partial(parallel_simulation.set_up_parallel_abc_sampler, **kwargs)
        filename = filename_base+'_m'+str(kwargs['dim_auxiliary_var'])
        filenames_list = [filename+'_'+str(k) for k in K_repetitions]
        partial_parallel_smc(filenames_list[0])
        functions_parallelize.parallelize_partial_over_chunks(partial_parallel_smc, filenames_list[1:])
