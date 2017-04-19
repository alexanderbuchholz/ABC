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

path = "/media/alex/Transcend/ABC_results_storage/simulation_results_18-4-17"
import gaussian_densities_etc
#import functions_tuberculosis_model as functions_model
import functions_mixture_model as functions_model
#import functions_mixture_model as functions_model
from class_smc import smc_sampler
import functions_propagate_reweight_resample

Time = 600
dim_particles = 1
repetitions = 4
N_particles = 10000
target_ESS_ratio_resampler = 0.3
target_ESS_ratio_reweighter = 0.3
epsilon_target = functions_model.epsilon_target(dim_particles) #0.001 #0.25

K_repetitions = range(repetitions)
#filename = functions_model.model_string+'_dim_'+str(dim_particles)+'_adaptive_M_autochoose_eps_gaussian_kernel'
filename = functions_model.model_string+'_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3'



#import plot_bivariate_scatter_hist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/alpha_stable_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/lotka_volterra_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
#import functions_tuberculosis_model as functions_mixture_model
#import functions_alpha_stable_model as functions_mixture_model
import functions_mixture_model as functions_mixture_model
#import functions_toggle_switch_model as functions_mixture_model
#import functions_lotka_volterra_model as functions_mixture_model
#import functions_mixture_model
model_description = functions_mixture_model.model_string

Time = 100
dim_auxiliary_var = 2
augment_M = False
M_incrementer = 2
target_ESS_ratio_reweighter = 0.5
target_ESS_ratio_resampler = 0.5
epsilon_target = functions_mixture_model.epsilon_target(dim_particles)
contracting_AIS = True
M_increase_until_acceptance = False
M_target_multiple_N = target_ESS_ratio_reweighter
covar_factor = 1.2
#propagation_mechanism = 'AIS'# AIS 'Del_Moral'#'nonparametric' #"true_sisson" neg_binomial
#sampler_type = 'QMC'
#y_simulation = 'neg_binomial' # 'standard' 'neg_binomial'
start_phase_ais = 5
truncate_neg_binomial = False
ancestor_sampling = "False" #"Hilbert"#False#"Hilbert"
resample = True
#autochoose_eps = 'quantile_based' # ''ess_based quantile_based
computational_budget = 10**6
parallelize = False
quantile_target = 0.3

propagation_mechanism = 'AIS'# AIS 'Del_Moral'#'nonparametric' #"true_sisson" neg_binomial
sampler_type = 'QMC'
y_simulation = 'neg_binomial' # 'standard' 'neg_binomial'
autochoose_eps = 'quantile_based' # ''ess_based quantile_based




model_description = model_description+'_'+sampler_type+'_'+propagation_mechanism+'_'+y_simulation
save = False
mixture_components = 10
kernel = gaussian_densities_etc.uniform_kernel
move_particle =gaussian_densities_etc.gaussian_move
y_star = functions_mixture_model.f_y_star(dim_particles)


test_sampler = smc_sampler(N_particles, 
                            dim_particles, 
                            Time, 
                            dim_auxiliary_var=dim_auxiliary_var, 
                            augment_M = augment_M,
                            M_incrementer = M_incrementer,  
                            ESS_treshold_resample=N_particles*(target_ESS_ratio_resampler), 
                            ESS_treshold_incrementer = N_particles*(target_ESS_ratio_reweighter),
                            epsilon_target=epsilon_target, 
                            contracting_AIS=contracting_AIS,
                            M_increase_until_acceptance=M_increase_until_acceptance,
                            M_target_multiple_N = M_target_multiple_N,
                            computational_budget = computational_budget,
                            y_simulation = y_simulation,
                            start_phase_ais = start_phase_ais, 
                            truncate_neg_binomial = truncate_neg_binomial,
                            quantile_target=quantile_target)


test_sampler.propagation_mechanism = propagation_mechanism
test_sampler.sampler_type = sampler_type
test_sampler.covar_factor = covar_factor


simulator_mm = functions_propagate_reweight_resample.simulator_sampler(functions_mixture_model.simulator,
                                    y_star,
                                    functions_mixture_model.delta,
                                    functions_mixture_model.exclude_theta,
                                    M_simulator = dim_auxiliary_var,
                                    parallelize = parallelize)
test_sampler.setAuxialiarySampler(simulator_mm)



propagater = functions_propagate_reweight_resample.propagater_particles(dim_particles,
                                                                        N_particles,
                                                                        move_particle,
                                                                        sampler_type=sampler_type,
                                                                        propagation_mechanism= propagation_mechanism,
                                                                        covar_factor = covar_factor,
                                                                        ancestor_sampling = ancestor_sampling,
                                                                        mixture_components = mixture_components)
test_sampler.setPropagateFunction(propagater.f_propagate)

reweighter = functions_propagate_reweight_resample.reweighter_particles(dim_particles,
                                                                        N_particles,
                                                                        propagation_mechanism= propagation_mechanism,
                                                                        covar_factor = covar_factor,
                                                                        autochoose_eps=autochoose_eps,
                                                                        target_ESS_ratio = target_ESS_ratio_reweighter,
                                                                        kernel = kernel, 
                                                                        epsilon_target = epsilon_target)
epsilon = np.linspace(10, epsilon_target, Time)
test_sampler.setEpsilonSchedule(epsilon)
test_sampler.setReweightFunction(reweighter.f_reweight)

test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_rqmc)
length_quantiles = 10
quantiles = np.linspace(0.1, 0.01, num=length_quantiles)

array_results_list = []
sampler_list = ['mc', 'qmc', 'rqmc']
repetitions = 40
for sampler in sampler_list:
    del test_sampler.f_initiate_particles
    #pdb.set_trace()
    if sampler == 'mc':
        test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_mc)
    elif sampler == 'qmc':
        test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_qmc)
    elif sampler == 'rqmc':
        test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_rqmc)
    else: raise ValueError('error in sampler!')
    array_results = np.zeros((2, length_quantiles, repetitions))
    for k_repetion in range(repetitions):
        test_sampler.accept_reject_sampler(10000)
        for j_quantile in range(length_quantiles):
            posterior = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), epsilon_target_accept_reject=quantiles[j_quantile])
            array_results[0, j_quantile, k_repetion] = posterior.mean()
            array_results[1, j_quantile, k_repetion] = posterior.var()
    array_results_list.append(array_results)

from matplotlib import pyplot as plt
plt.hist(array_results_list[0][0,0,:]); plt.show()
pdb.set_trace()

plt.title("Variance of the variance estimator")
plt.plot(quantiles, array_results_list[0].var(axis=2)[1,:])
plt.plot(quantiles, array_results_list[1].var(axis=2)[1,:])
plt.plot(quantiles, array_results_list[2].var(axis=2)[1,:])
plt.xlabel('Quantile of distance'); plt.ylabel('Variance of estimator')
plt.legend(sampler_list)
plt.savefig("variance_of_variance_estimator.png")
plt.show()


plt.title("Variance of the mean estimator")
plt.plot(quantiles, array_results_list[0].var(axis=2)[0,:])
plt.plot(quantiles, array_results_list[1].var(axis=2)[0,:])
plt.plot(quantiles, array_results_list[2].var(axis=2)[0,:])
plt.xlabel('Quantile of distance'); plt.ylabel('Variance of estimator')
plt.legend(sampler_list)
plt.savefig("variance_of_mean_estimator.png")
plt.show()

# posterior qmc
test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_qmc)
test_sampler.accept_reject_sampler(100000)
posterior_qmc = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), epsilon_target_accept_reject=0.01)

test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_mc)
test_sampler.accept_reject_sampler(100000)
posterior_mc = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), epsilon_target_accept_reject=0.01)

plt.title("Posterior distribution for epsilon = 0.01")
sns.distplot(posterior_mc.flatten(), label="mc posterior")
sns.distplot(posterior_qmc.flatten(), label="qmc posterior")
plt.savefig("posterior_distribution_mixed_gaussian.png")
plt.show()

#import yappi
#yappi.start()
#test_sampler.iterate_smc(resample=resample, save=save, modified_sampling=propagation_mechanism)
#yappi.get_func_stats().print_all()

pdb.set_trace()
pdb.set_trace()


