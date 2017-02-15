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
path1 = "/home/alex/python_programming/ABC_results_storage/simulation_results_14-2-17"
path2 = "/home/alex/python_programming/ABC_results_storage/simulation_results"
import os
os.chdir(path2)
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/simulation")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
#import sisson_simulation_parameters_mixture_model
#import simulation_parameters_mixture_model_3_2_17 as simulation_parameters_model
import simulation_parameters_mixture_model_15_2_17 as simulation_parameters_model
#import a17_1_17_sisson_simulation_parameters_tuberculosis_model as sisson_simulation_parameters_mixture_model
#import a20_1_17_simulation_parameters_tuberculosis_model as simulation_parameters_mixture_model
import f_rand_seq_gen
import gaussian_densities_etc
def f_summary_stats(parameters, sample_method = "MC", particles=500, propagation_method = 'AIS', cum_sum=False):
    #parameters.repetitions = 2
    #pdb.set_trace()
    final_means = np.zeros((parameters.kwargs["dim_particles"], parameters.repetitions))
    final_ESS = np.zeros((1, parameters.repetitions))
    final_epsilon = np.zeros((1, parameters.repetitions))
    final_simulation_time = np.zeros((1, parameters.repetitions))
    final_number_simulations = np.zeros((1, parameters.repetitions))
    number_simulations = np.zeros((1, parameters.repetitions, parameters.kwargs["Time"]))

    means = np.zeros((parameters.kwargs["dim_particles"], parameters.Time, parameters.repetitions))
    epsilons = np.zeros((1, parameters.Time, parameters.repetitions))
    for i_simulation in range(parameters.repetitions):
    #for i_simulation in range(32):
        if propagation_method == 'true_sisson':
            simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(1)+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ) )
        else:
            simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(parameters.kwargs["dim_auxiliary_var"])+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ) )
        #pdb.set_trace()
        selector = simulation["means_particles"].shape[1] #np.min((simulation['T_max'], parameters.Time))
        #pdb.set_trace()
        #if propagation_method == 'Del_Moral':
        #    selector = selector - 1
        #pdb.set_trace()
        final_means[:, i_simulation] = simulation["means_particles"][:, selector-1] # TODO: error here ? is the range correct?
        means[:, :selector, i_simulation] = simulation["means_particles"][:, :selector]
        final_ESS[:,i_simulation] = simulation["ESS"][selector-1]
        final_epsilon[:,i_simulation] = simulation["epsilon"][selector-1]
        epsilons[:,:len(simulation["epsilon"]),i_simulation] = simulation["epsilon"]
        final_simulation_time[:,i_simulation] = simulation["simulation_time"]
        if propagation_method == 'AIS':
            final_number_simulations[:,i_simulation]=sum(simulation["M_list"])*simulation['N']
            number_simulations[0,i_simulation, :selector] = np.array(simulation["M_list"])
            if cum_sum==True:
                number_simulations[0,i_simulation, :selector] = np.cumsum(number_simulations[0,i_simulation, :selector])
        elif propagation_method == 'Del_Moral':
            final_number_simulations[:,i_simulation]= simulation["M"]*simulation['N']*simulation['T_max']
            number_simulations[0,i_simulation, :selector] = np.array(simulation["M"])
            if cum_sum==True:
                number_simulations[0,i_simulation, :selector] = np.cumsum(number_simulations[0,i_simulation, :selector])
        else:
            final_number_simulations[:,i_simulation]= simulation['sampling_counter']
            number_simulations[0,i_simulation, :selector] = np.array(simulation["M_list"])/simulation['N']
            if cum_sum==True:
                number_simulations[0,i_simulation, :selector] = np.cumsum(number_simulations[0,i_simulation, :selector])
            #pdb.set_trace()
        
    #pdb.set_trace()
    means_all = means[:, :selector, :]
    epsilons = epsilons[:,:len(simulation["epsilon"]),:]
    var_all = means_all.var(axis=2)
    means_last = np.nanmean(final_means, axis=1)
    #means_var = np.nanvar(final_means, axis=1)
    means_var_last = (final_means**2).mean(axis=1) # use this for the MSE
    ESS_mean = final_ESS.mean()
    epsilon_mean = final_epsilon.mean()
    time_mean = final_simulation_time.mean()
    number_simulations_mean = final_number_simulations.mean()
    number_simulations = number_simulations[0,:, :selector]
    #pdb.set_trace()
    return [means_last, means_var_last, ESS_mean, epsilon_mean, time_mean, number_simulations_mean], [means_all, epsilons, means_all.var(axis=2), number_simulations]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#pdb.set_trace()
#sisson_simulation_results = f_summary_stats(sisson_simulation_parameters_mixture_model, sample_method = "MC", particles=100)
if True:
    var_different_methods = np.zeros((5,len(simulation_parameters_model.kwargs['N_particles_list']), simulation_parameters_model.kwargs['Time']))
    counter = 0
    for N_particles in simulation_parameters_model.kwargs['N_particles_list']:
        QMC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "QMC", particles=N_particles, propagation_method = 'AIS')
        MC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'AIS')
        RQMC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "RQMC", particles=N_particles, propagation_method = 'AIS')
        del_moral_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'Del_Moral')
        true_sisson_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'true_sisson')
        #print sisson_simulation_results[0]
        
        print N_particles
        print MC_simulation_results[0]
        print QMC_simulation_results[0]
        print RQMC_simulation_results[0]
        print del_moral_simulation_results[0]
        print true_sisson_simulation_results[0]
        #pdb.set_trace()
    """
        for i_epsilon in range(QMC_simulation_results[1][1].shape[1]):
            var_different_methods[0,counter, i_epsilon] = MC_simulation_results[1][2][0,i_epsilon]
            var_different_methods[1,counter, i_epsilon] = QMC_simulation_results[1][2][0,i_epsilon]
            var_different_methods[2,counter, i_epsilon] = RQMC_simulation_results[1][2][0,i_epsilon]
            var_different_methods[3,counter] = del_moral_simulation_results[0][1][0]
            var_different_methods[4,counter] = true_sisson_simulation_results[0][1][0]
        counter += 1
        print '\n'
    for i_epsilon in  range(QMC_simulation_results[1][1].shape[1]): #[QMC_simulation_results[1][1].shape[1]-1]: #
        sns.set_style("darkgrid")
        #pdb.set_trace()
        plt.title('MSE for '+simulation_parameters_model.functions_model.model_string+' epsilon: '+str(QMC_simulation_results[1][1][:,i_epsilon,0]))
        plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[0,:,i_epsilon], label="MC")
        plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[1,:,i_epsilon], label="QMC")
        plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[2,:,i_epsilon], label="RQMC")
        plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[3,:,i_epsilon], label="Del Moral")
        plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[4,:,i_epsilon], label="Sisson")
        plt.yscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('N')
        plt.ylabel('variance')
        plt.savefig(str(i_epsilon)+'N_variance.png')
        plt.show()
        plt.close()
    """
#simulation = pickle.load( open( "mixture_gaussians_diff_variance_adaptive_M_autochoose_eps_gaussian_kernel10_MC10_AIS_2500_simulation_abc_epsilon_0.025_40.p", "rb" ) )

def function_flatten_results(_results, dim):
    #pdb.set_trace()
    _means_inter = _results[1][0][dim, :, :].flatten()
    _epsilons_inter = _results[1][1].flatten()
    _means_inter = _means_inter[_epsilons_inter > 0.]
    _epsilons_inter = _epsilons_inter[_epsilons_inter > 0.]
    return _means_inter, _epsilons_inter

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
        #pdb.set_trace()
        print('code works for one dimension only!')
        MC_means_inter, MC_epsilons_inter = function_flatten_results(MC_results, 0)
        QMC_means_inter, QMC_epsilons_inter = function_flatten_results(QMC_results, 0)
        RQMC_means_inter, RQMC_epsilons_inter = function_flatten_results(RQMC_results, 0)
        Del_Moral_means_inter, Del_Moral_epsilons_inter = function_flatten_results(Del_Moral_results, 0)
        Sisson_means_inter, Sisson_epsilons_inter = function_flatten_results(Sisson_results, 0)

        sns.set_style("darkgrid")
        #sns.tsplot(time=MC_epsilons_inter, data=MC_means_inter, color='blue')
        #sns.tsplot(time=RQMC_epsilons_inter, data=RQMC_means_inter, color='green')
        #sns.tsplot(time=Del_Moral_epsilons_inter, data=Del_Moral_means_inter, color='red')
        #plt.subplot(1,1,1)
        #pdb.set_trace()
        
        plt.title('means and epsilon for N:'+str(N_particles))
        plt.scatter(MC_epsilons_inter, MC_means_inter, lw=0.5, alpha=1, color='blue', label="MC")
        plt.scatter(QMC_epsilons_inter, QMC_means_inter, lw=0.5, alpha=1, color='cyan', label="QMC")
        plt.scatter(RQMC_epsilons_inter, RQMC_means_inter, lw=0.5, alpha=1, color='green', label="RQMC")
        plt.scatter(Del_Moral_epsilons_inter, Del_Moral_means_inter, color='red', label='Del Moral')
        plt.scatter(Sisson_epsilons_inter, Sisson_means_inter, color='yellow', label='Sisson')
        #plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('means')
        plt.savefig(str(N_particles)+'N_means_epsilon.png')
        plt.show()

        #pdb.set_trace()
        plt.title('MSE for '+simulation_parameters_model.functions_model.model_string+' over epsilon and N:'+str(N_particles))
        plt.plot(MC_results[1][1][0,:,0], MC_results[1][2][0,:]*MC_results[1][3].mean(axis=0), label="MC")
        plt.plot(QMC_results[1][1][0,:,0], QMC_results[1][2][0,:]*QMC_results[1][3].mean(axis=0), label="QMC")
        #pdb.set_trace()
        plt.plot(RQMC_results[1][1][0,:,0], RQMC_results[1][2][0,:]*RQMC_results[1][3].mean(axis=0), label="RQMC")
        plt.plot(Del_Moral_results[1][1][0,:-1,0], Del_Moral_results[1][2][0,:-1]*Del_Moral_results[1][3][:,:-1].mean(axis=0), label="Del Moral")
        plt.plot(Sisson_results[1][1][0,:,0], Sisson_results[1][2][0,:]*Sisson_results[1][3].mean(axis=0), label="Sisson")
        #plt.plot(simulation_parameters_model.kwargs['N_particles_list'], var_different_methods[4,:], label="Sisson")
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
        plt.xlabel('epsilon')
        plt.ylabel('variance')
        plt.savefig(str(N_particles)+'N_variance_epsilon.png')
        plt.show()


        #pdb.set_trace()
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
os.chdir(path2)
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
