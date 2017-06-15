# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:21:52 2017
    evaluation of results
@author: alex
"""

from functions_evaluation import *

#import sisson_simulation_parameters_mixture_model
#import simulation_parameters_mixture_model_3_2_17 as simulation_parameters_model
#import simulation_parameters_mixture_model_17_2_17 as simulation_parameters_model
#import a17_1_17_sisson_simulation_parameters_tuberculosis_model as sisson_simulation_parameters_mixture_model
#import simulation_parameters_mixture_model_17_2_17 as simulation_parameters_model
#import simulation_parameters_mixture_model_mixed_gaussian_dim2_18_4_17_desktop as simulation_parameters_model

#import simulation_parameters_mixture_model_mixed_gaussian_dim2_16_5_17_desktop as simulation_parameters_model
import simulation_parameters_mixture_model_mixed_gaussian_dim1_14_6_17_desktop as simulation_parameters_model
path = simulation_parameters_model.path
os.chdir(path)


if True:
    N_particles_list = simulation_parameters_model.kwargs['N_particles_list']
    MC_means = []
    RQMC_means = []
    cum_sum = True
    for N_particles in N_particles_list:
        MC_results =  f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, cum_sum=cum_sum)
        QMC_results =  f_summary_stats(simulation_parameters_model, sample_method = "QMC", particles=N_particles, cum_sum=cum_sum)
        RQMC_results = f_summary_stats(simulation_parameters_model, sample_method = "RQMC", particles=N_particles, cum_sum=cum_sum)
        Del_Moral_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'Del_Moral', cum_sum=cum_sum)
        Sisson_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'true_sisson', cum_sum=cum_sum)

        results_summary_to_save = {}
        results_summary_to_save['MC'] = MC_results
        results_summary_to_save['QMC'] = QMC_results
        results_summary_to_save['RQMC'] = RQMC_results
        results_summary_to_save['Del_Moral'] = Del_Moral_results
        results_summary_to_save['Sisson'] = Sisson_results
        import pickle
        pickle.dump(results_summary_to_save, open(str(N_particles)+"_results_summary_to_save.p", "wb" ))