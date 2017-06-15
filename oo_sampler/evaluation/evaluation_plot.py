# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:21:52 2017
    evaluation of results
@author: alex
"""
# pylint: disable=C0321
import ipdb as pdb
import pickle
import numpy as np

from functions_evaluation import *
#if __name__ == '__main__':
#path1 = "/home/alex/python_programming/ABC_results_storage/simulation_results_18-4-17"
#path2 = "/home/alex/python_programming/ABC_results_storage/simulation_results"
import os
root_path = "/home/alex/python_programming"
import sys
sys.path.append(root_path+"/ABC/oo_sampler/simulation")
sys.path.append(root_path+"/ABC/oo_sampler/functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")
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
#import simulation_parameters_mixture_model_single_gaussian_dim3_30_3_17_desktop as simulation_parameters_model
import f_rand_seq_gen
import gaussian_densities_etc

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




if True:
    N_particles_list = simulation_parameters_model.kwargs['N_particles_list']
    for N_particles in N_particles_list:
        results_summary_to_save = pickle.load(open( str(N_particles)+"_results_summary_to_save.p", "rb" ) )
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

        sns.set_style("darkgrid")
        if False: 
            MC_means_inter, MC_epsilons_inter = function_flatten_results(MC_results, 0)
            QMC_means_inter, QMC_epsilons_inter = function_flatten_results(QMC_results, 0)
            RQMC_means_inter, RQMC_epsilons_inter = function_flatten_results(RQMC_results, 0)
            Del_Moral_means_inter, Del_Moral_epsilons_inter = function_flatten_results(Del_Moral_results, 0, method="Del_Moral")
            #Sisson_means_inter, Sisson_epsilons_inter = function_flatten_results(Sisson_results, 0)

            plt.title('means and epsilon for N:'+str(N_particles))
            #plt.scatter(MC_epsilons_inter, MC_means_inter, lw=0.5, alpha=1, color='blue', label="MC")
            plt.scatter(QMC_epsilons_inter, QMC_means_inter, lw=0.5, alpha=1, color='cyan', label="QMC")
            plt.scatter(RQMC_epsilons_inter, RQMC_means_inter, lw=0.5, alpha=1, color='green', label="RQMC")
            plt.scatter(Del_Moral_epsilons_inter, Del_Moral_means_inter, color='red', label='Del Moral')
            #plt.scatter(Sisson_epsilons_inter, Sisson_means_inter, color='yellow', label='Sisson')
            #plt.yscale('log')
            plt.xscale('log')
            plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
            plt.xlabel('epsilon')
            plt.ylabel('means')
            #plt.savefig(str(N_particles)+'N_means_epsilon.png')
            plt.show()


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


        plt.title('MSE of variance for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
        plot_no_double_epsilon_variance(MC_results, 'MC', true_variance= simulation_parameters_model.functions_model.var)
        plot_no_double_epsilon_variance(QMC_results, 'QMC', true_variance= simulation_parameters_model.functions_model.var)
        plot_no_double_epsilon_variance(RQMC_results, 'RQMC', true_variance= simulation_parameters_model.functions_model.var)
        plot_no_double_epsilon_variance(Del_Moral_results, 'Del Moral', true_variance= simulation_parameters_model.functions_model.var)
        plot_no_double_epsilon_variance(Sisson_results, 'Sisson', true_variance= simulation_parameters_model.functions_model.var)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('MSE times cumulative budget')
        plt.savefig('mse_variance_budget'+str(N_particles)+'N_variance_epsilon.png')
        plt.show()

        pdb.set_trace()
        plt.title('MSE for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N = '+str(N_particles))
        plot_no_double_epsilon(MC_results, 'MC')
        plot_no_double_epsilon(QMC_results, 'QMC')
        plot_no_double_epsilon(RQMC_results, 'RQMC')
        plot_no_double_epsilon(Del_Moral_results, 'Del Moral')
        plot_no_double_epsilon(Sisson_results, 'Sisson')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('MSE times cumulative budget')
        plt.savefig('mse_cum_budget'+str(N_particles)+'N_variance_epsilon.png')
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
