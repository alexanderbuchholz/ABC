# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:21:52 2017
    evaluation of results
@author: alex
"""
import ipdb as pdb
import pickle
import numpy as np
#if __name__ == '__main__':
path1 = "/home/alex/python_programming/ABC_results_storage/simulation_results"
path2 = "/home/alex/python_programming/ABC_results_storage/simulation_results"
import os
os.chdir(path1)
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/simulation")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
#import sisson_simulation_parameters_mixture_model
#import simulation_parameters_mixture_model_3_2_17 as simulation_parameters_model
import simulation_parameters_mixture_model_5_2_17 as simulation_parameters_model
#import a17_1_17_sisson_simulation_parameters_tuberculosis_model as sisson_simulation_parameters_mixture_model
#import a20_1_17_simulation_parameters_tuberculosis_model as simulation_parameters_mixture_model
import f_rand_seq_gen
import gaussian_densities_etc
def f_summary_stats(parameters, sample_method = "MC", particles=500, propagation_method = 'AIS'):
    #parameters.repetitions = 2
    #pdb.set_trace()
    final_means = np.zeros((parameters.kwargs["dim_particles"], parameters.repetitions))
    final_ESS = np.zeros((1, parameters.repetitions))
    final_epsilon = np.zeros((1, parameters.repetitions))
    final_simulation_time = np.zeros((1, parameters.repetitions))
    final_number_simulations = np.zeros((1, parameters.repetitions))

    means = np.zeros((parameters.kwargs["dim_particles"], parameters.Time, parameters.repetitions))
    epsilons = np.zeros((1, parameters.Time, parameters.repetitions))
    for i_simulation in range(parameters.repetitions):
        simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(parameters.kwargs["dim_auxiliary_var"])+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+'_'+str(parameters.Time)+".p", "rb" ) )
        final_means[:, i_simulation] = simulation["means_particles"][:, -1]
        #pdb.set_trace()
        means[:, :simulation['T_max'], i_simulation] = simulation["means_particles"]
        final_ESS[:,i_simulation] = simulation["ESS"][-1]
        final_epsilon[:,i_simulation] = simulation["epsilon"][-1]
        epsilons[:,:len(simulation["epsilon"]),i_simulation] = simulation["epsilon"]
        final_simulation_time[:,i_simulation] = simulation["simulation_time"]
        if propagation_method == 'AIS':
            #pdb.set_trace()
            final_number_simulations[:,i_simulation]=sum(simulation["M_list"])*simulation['N']
        elif propagation_method == 'Del_Moral':
            final_number_simulations[:,i_simulation]= simulation["M"]*simulation['N']*simulation['T_max']
        else:
            final_number_simulations[:,i_simulation]= simulation['sampling_counter']
    #pdb.set_trace()
    means_means = np.nanmean(final_means, axis=1)
    means_var = np.nanvar(final_means, axis=1)
    #means_var = (final_means**2).mean(axis=1) # use this for the MSE
    ESS_mean = final_ESS.mean()
    epsilon_mean = final_epsilon.mean()
    time_mean = final_simulation_time.mean()
    number_simulations_mean = final_number_simulations.mean()
    return [means_means, means_var, ESS_mean, epsilon_mean, time_mean, number_simulations_mean], [means, epsilons]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pdb.set_trace()
#sisson_simulation_results = f_summary_stats(sisson_simulation_parameters_mixture_model, sample_method = "MC", particles=100)
if False:
    for N_particles in simulation_parameters_model.kwargs['N_particles_list']:
        MC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'AIS')
        RQMC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "RQMC", particles=N_particles, propagation_method = 'AIS')
        del_moral_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'Del_Moral')
    #print sisson_simulation_results[0]
        print MC_simulation_results[0]
        print RQMC_simulation_results[0]
        print del_moral_simulation_results[0]
        print '\n'
    pdb.set_trace()
if True:
    N_particles_list = simulation_parameters_model.kwargs['N_particles_list']
    MC_var = []
    RQMC_var = []
    for N_particles in N_particles_list:
        MC_results =  f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles)
        RQMC_results = f_summary_stats(simulation_parameters_model, sample_method = "RQMC", particles=N_particles)
        pdb.set_trace()
        MC_var.append(MC_results[1].mean())
        RQMC_var.append(RQMC_results[1].mean())
    sns.set_style("darkgrid")
    plt.subplot(1,1,1)
    pdb.set_trace()
    plt.plot(N_particles_list, MC_var, color='blue', lw=2)
    plt.plot(N_particles_list, RQMC_var, color='green', lw=2)
    plt.yscale('log')
    plt.show()
    plt.plot(N_particles_list, np.array(MC_var)/np.array(RQMC_var), color='blue', lw=2)
    plt.show()
pdb.set_trace()

if False:
    dim = 1
    mean_values_sisson = sisson_simulation_results[1][0]
    epsilon_values_sisson = sisson_simulation_results[1][1]
    mean_values_sisson[mean_values_sisson==0.]=np.nan


#    mean_values_MC = MC_simulation_results[1][0]
#    epsilon_values_MC = MC_simulation_results[1][1]
#    mean_values_MC[mean_values_MC==0.]=np.nan

    mean_values_RQMC = RQMC_simulation_results[1][0]
    epsilon_values_RQMC = RQMC_simulation_results[1][1]
    mean_values_RQMC[mean_values_RQMC==0.]=np.nan
    fig = plt.figure()
    fig.suptitle('The mean of the parameters and the epsilon schedule', fontsize=14)
    plt.xlabel('epsilon schedule', fontsize=14)
    plt.ylabel('mean of posterior parameter', fontsize=14)
    sisson_plot  = plt.scatter(epsilon_values_sisson.flatten(), mean_values_sisson[dim,:,:].flatten(), color='r',marker='^', alpha=.4)
#    mc_plot = plt.scatter(epsilon_values_MC.flatten(), mean_values_MC[dim,:,:].flatten(), color='b', alpha=.4)
    rqmc_plot = plt.scatter(epsilon_values_RQMC.flatten(), mean_values_RQMC[dim,:,:].flatten(), color='g', marker='s', alpha=.4)
#    plt.legend((sisson_plot, mc_plot, rqmc_plot),
#           ('Sisson', 'AIS MC', 'AIS RQMC'),
#           scatterpoints=1,
#           loc='lower left',
#           ncol=3,
#           fontsize=14)
    plt.show()
    pdb.set_trace()

def resample_for_plotting(particles, weights):
    particles_resampled = np.zeros(particles.shape)
    N_particles = particles.shape[1]
    u_new =f_rand_seq_gen.random_sequence_mc(1, i=0, n=N_particles)
    for i_particle in range(N_particles):
        # resampling to get the ancestors
        # TODO: Implement Hilber sampling
        ancestor_new = gaussian_densities_etc.weighted_choice( weights[0,:], u_new[i_particle]) # get the ancestor
        particles_resampled[:,i_particle] = particles[:, ancestor_new] # save the particles
    # resampling done, update the weights
    gaussian_densities_etc.break_if_nan(particles_resampled)
    #pdb.set_trace()
    return particles_resampled


def f_accept_reject_precalculated_particles(precalculated_particles, precalculated_auxialiary_particles, epsilon_target_accept_reject):
    accept_reject_selector = precalculated_auxialiary_particles < epsilon_target_accept_reject
    return precalculated_particles[:, accept_reject_selector]

simulation_RQMC = pickle.load( open("mixture_gaussians_diff_variance_adaptive_M_autochoose_eps_gaussian_kernel1_RQMC1_AIS_4000_simulation_abc_epsilon_40.p", "rb"))
simulation_MC = pickle.load( open("mixture_gaussians_diff_variance_adaptive_M_autochoose_eps_gaussian_kernel0_RQMC1_true_sisson_500_simulation_abc_epsilon_0.025_40.p", "rb"))
os.chdir(path2)
#simulation_sisson = pickle.load( open("tuberculosis_true_sission17_MC1_AIS_200_simulation_abc_epsilon_24.p", "rb"))

#simulation_MC = pickle.load( open("adaptive_M_autochoose_eps_uniform_kernel10_MC2_AIS_5000_simulation_abc_epsilon_30.p", "rb"))
#simulation_MC = pickle.load( open( filename+str(1)+"_"+"MC"+"_AIS_"+str(500)+"_simulation_abc_epsilon_"+str(parameters.repetitions)+".p", "rb" ) )
#simulation_RQMC = pickle.load( open( filename+str(1)+"_"+"RQMC"+"_AIS_"+str(500)+"_simulation_abc_epsilon_"+str(parameters.repetitions)+".p", "rb" ) )
#pdb.set_trace()
#sisson_resampled = resample_for_plotting(simulation_sisson['particles'][:,:,-1], simulation_sisson['weights'][:,:,-1])

rqmc_resampled = resample_for_plotting(simulation_RQMC['particles'][:,:,simulation_RQMC["T_max"]-1], simulation_RQMC['weights'][:,:,simulation_RQMC["T_max"]-1])
mc_resampled = resample_for_plotting(simulation_MC['particles'][:,:,simulation_MC["T_max"]-1], simulation_MC['weights'][:,:,simulation_MC["T_max"]-1])

#pdb.set_trace()
#x1_sisson = pd.Series(sisson_resampled[0,:], name="$X_1$")
#x2_sisson = pd.Series(sisson_resampled[1,:], name="$X_2$")
#sns.jointplot(x1_sisson, x2_sisson, kind="kde")


#pdb.set_trace()
precomputed_data = simulation_parameters_model.functions_model.load_precomputed_data(simulation_parameters_model.dim_particles, simulation_parameters_model.functions_model.exponent)
precalculated_particles = precomputed_data['theta_values']
precalculated_auxialiary_particles = precomputed_data['y_diff_values']
#pdb.set_trace()
AR_posterior_particles = f_accept_reject_precalculated_particles(precalculated_particles, precalculated_auxialiary_particles,  simulation_parameters_model.epsilon_target)

g = sns.distplot(AR_posterior_particles[0,:], label="AR")
plt.subplots_adjust(top=0.9)
plt.title(('epsilon = %s, \n  N_AR = %d, N = %d')% (simulation_parameters_model.epsilon_target, AR_posterior_particles.shape[1], simulation_RQMC['N']))
sns.kdeplot(rqmc_resampled.flatten(), label="RQMC")
sns.kdeplot(mc_resampled.flatten(), label="MC")
plt.show()

sum(simulation_RQMC['M_list'])*simulation_RQMC['N']
#sum(simulation_MC['M_list'])*simulation_MC['N']
#simulation_MC['means_particles']
#x1_rqmc = pd.Series(rqmc_resampled[0,:], name="$X_1$")
#x2_rqmc = pd.Series(rqmc_resampled[1,:], name="$X_2$")
#sns.jointplot(x1_rqmc, x2_rqmc, kind="kde")
#x1_mc = pd.Series(mc_resampled[0,:], name="$X_1$")
#x2_mc = pd.Series(mc_resampled[1,:], name="$X_2$")
#sns.jointplot(x1_mc, x2_mc, kind="kde")


pdb.set_trace()