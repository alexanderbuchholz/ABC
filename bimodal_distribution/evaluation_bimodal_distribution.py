# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:21:52 2017
    evaluation of results
@author: alex
"""
import ipdb as pdb
import pickle
import numpy as np

import os
root_path = "/home/alex/python_programming"
import sys
sys.path.append(root_path+"/ABC/oo_sampler/simulation")
sys.path.append(root_path+"/ABC/oo_sampler/functions")
sys.path.append(root_path+"/ABC/oo_sampler/evaluation")
sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")

import simulation_parameters_mixture_model_bimodal_gaussian_dim2_11_6_17_desktop as simulation_parameters_model
from functions_evaluation import *

path_current = os.getcwd()

path = simulation_parameters_model.path
os.chdir(path)

import f_rand_seq_gen
import gaussian_densities_etc

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


name_qmc = 'mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_4_QMC2_AIS_500_simulation_abc_epsilon_0.005.p'
posterior_plot = pickle.load(open(name_qmc, "rb"))
sns.kdeplot(posterior_plot['particles'][0,:,-1], posterior_plot['particles'][1,:,-1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('posterior_bimodal.png')
plt.show()


pdb.set_trace()

if False:
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



N_particles_list = simulation_parameters_model.kwargs['N_particles_list']
for N_particles in N_particles_list:
    results_summary_to_save = pickle.load(open( str(N_particles)+"_results_summary_to_save.p", "rb" ) )

    os.chdir(path_current)
    #pdb.set_trace()
    MC_results = results_summary_to_save['MC']
    QMC_results = results_summary_to_save['QMC']
    RQMC_results = results_summary_to_save['RQMC']
    Del_Moral_results = results_summary_to_save['Del_Moral']
    Sisson_results = results_summary_to_save['Sisson']

    print simulation_parameters_model.filename
    print N_particles
    print MC_results[0]
    print QMC_results[0]
    print RQMC_results[0]
    print Del_Moral_results[0]
    print Sisson_results[0]
    pdb.set_trace()
    sns.set_style("darkgrid")


    if False:
        plt.title('ESS for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
        plot_no_double_epsilon_ESS(MC_results, 'MC')
        plot_no_double_epsilon_ESS(QMC_results, 'QMC')
        plot_no_double_epsilon_ESS(RQMC_results, 'RQMC')
        plot_no_double_epsilon_ESS(Del_Moral_results, 'Del Moral')
        plot_no_double_epsilon_ESS(Sisson_results, 'Sisson')
        #plt.yscale('log')
        plt.xscale('log')
        plt.legend( numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('ESS')
        plt.savefig('ESS_'+str(N_particles)+'N_variance_epsilon.png')
        plt.show()

    if False:
        plt.title('L1 distance for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
        plot_no_double_epsilon_l1_distance(MC_results, 'MC')
        plot_no_double_epsilon_l1_distance(QMC_results, 'QMC')
        plot_no_double_epsilon_l1_distance(RQMC_results, 'RQMC')
        plot_no_double_epsilon_l1_distance(Del_Moral_results, 'Del Moral')
        plot_no_double_epsilon_l1_distance(Sisson_results, 'Sisson')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('L1 distance to true posterior')
        plt.savefig('l1distance_'+str(N_particles)+'N_variance_epsilon.png')
        plt.show()

    if True:
        #plt.title('MSE of variance for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
        plot_no_double_epsilon_variance_simple(MC_results, 'MC')
        plot_no_double_epsilon_variance_simple(QMC_results, 'QMC')
        plot_no_double_epsilon_variance_simple(RQMC_results, 'RQMC')
        plot_no_double_epsilon_variance_simple(Del_Moral_results, 'Del Moral')
        plot_no_double_epsilon_variance_simple(Sisson_results, 'Sisson')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon', fontsize=14)
        plt.ylabel('Variance times cumulative budget', fontsize=14)
        plt.savefig('variance_variance_budget'+str(N_particles)+'N_variance_epsilon.png')
        plt.show()

    pdb.set_trace()
    #plt.title('MSE for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
    plot_no_double_epsilon(MC_results, 'MC')
    plot_no_double_epsilon(QMC_results, 'QMC')
    plot_no_double_epsilon(RQMC_results, 'RQMC')
    plot_no_double_epsilon(Del_Moral_results, 'Del Moral')
    plot_no_double_epsilon(Sisson_results, 'Sisson')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
    plt.xlabel('epsilon', fontsize=14)
    plt.ylabel('Variance times cumulative budget', fontsize=14)
    plt.savefig('variance_mean_cum_budget'+str(N_particles)+'N_variance_epsilon.png')
    plt.show()


    plot_no_double_epsilon_number_simulations(MC_results, 'MC')
    plot_no_double_epsilon_number_simulations(QMC_results, 'QMC')
    plot_no_double_epsilon_number_simulations(RQMC_results, 'RQMC')
    plot_no_double_epsilon_number_simulations(Del_Moral_results, 'Del Moral')
    plot_no_double_epsilon_number_simulations(Sisson_results, 'Sisson')
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
    plt.xlabel('epsilon', fontsize=14)
    plt.ylabel('Cumulative number of simulations', fontsize=14)
    plt.savefig('number_simulations_cum_budget'+str(N_particles)+'N_variance_epsilon.png')
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
    import resampling
    #u_new =f_rand_seq_gen.random_sequence_mc(1, i=0, n=N_particles)
    #for i_particle in range(N_particles):
    #    # resampling to get the ancestors
    #    # TODO: Implement Hilber sampling
    #    ancestor_new = gaussian_densities_etc.weighted_choice( weights[0,:], u_new[i_particle]) # get the ancestor
    #    particles_resampled[:,i_particle] = particles[:, ancestor_new] # save the particles
    # resampling done, update the weights
    #pdb.set_trace()
    ancestors = resampling.stratified_resample(weights.squeeze())
    particles_resampled = particles[:, ancestors]
    gaussian_densities_etc.break_if_nan(particles_resampled)
    #pdb.set_trace()
    return particles_resampled


def f_accept_reject_precalculated_particles(precalculated_particles, precalculated_auxialiary_particles, epsilon_target_accept_reject):
    accept_reject_selector = precalculated_auxialiary_particles < epsilon_target_accept_reject
    return precalculated_particles[:, accept_reject_selector]

simulation_RQMC = pickle.load( open("mixture_gaussians_diff_variance_adaptive_M_autochoose_eps_gaussian_kernel_1_VB_component14_RQMC10_AIS_750_simulation_abc_epsilon_0.001_60.p", "rb"))
simulation_MC = pickle.load( open("mixture_gaussians_diff_variance_adaptive_M_autochoose_eps_gaussian_kernel_1_VB_component14_MC10_AIS_750_simulation_abc_epsilon_0.001_60.p", "rb"))
#os.chdir(path2)

#simulation_sisson = pickle.load( open("tuberculosis_true_sission17_MC1_AIS_200_simulation_abc_epsilon_24.p", "rb"))

#simulation_MC = pickle.load( open("adaptive_M_autochoose_eps_uniform_kernel10_MC2_AIS_5000_simulation_abc_epsilon_30.p", "rb"))
#simulation_MC = pickle.load( open( filename+str(1)+"_"+"MC"+"_AIS_"+str(500)+"_simulation_abc_epsilon_"+str(parameters.repetitions)+".p", "rb" ) )
#simulation_RQMC = pickle.load( open( filename+str(1)+"_"+"RQMC"+"_AIS_"+str(500)+"_simulation_abc_epsilon_"+str(parameters.repetitions)+".p", "rb" ) )
#pdb.set_trace()
#sisson_resampled = resample_for_plotting(simulation_sisson['particles'][:,:,-1], simulation_sisson['weights'][:,:,-1])

rqmc_resampled = resample_for_plotting(simulation_RQMC['particles'][:,:,simulation_RQMC["T_max"]], simulation_RQMC['weights'][:,:,simulation_RQMC["T_max"]])
mc_resampled = resample_for_plotting(simulation_MC['particles'][:,:,simulation_MC["T_max"]], simulation_MC['weights'][:,:,simulation_MC["T_max"]])

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
