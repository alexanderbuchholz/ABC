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
def f_summary_stats(parameters, sample_method="MC", particles=500, propagation_method = 'AIS', cum_sum=False):
    #parameters.repetitions = 2
    #pdb.set_trace()
    final_means = np.zeros((parameters.kwargs["dim_particles"], parameters.repetitions))
    final_ESS = np.zeros((1, parameters.repetitions))
    final_epsilon = np.zeros((1, parameters.repetitions))
    final_simulation_time = np.zeros((1, parameters.repetitions))
    final_number_simulations = np.zeros((1, parameters.repetitions))
    number_simulations = np.zeros((1, parameters.repetitions, parameters.kwargs["Time"]))

    means = np.zeros((parameters.kwargs["dim_particles"], parameters.Time, parameters.repetitions))
    l1_distances = np.zeros((1, parameters.Time, parameters.repetitions))
    vars = np.zeros((parameters.kwargs["dim_particles"], parameters.Time, parameters.repetitions))
    epsilons = np.zeros((1, parameters.Time, parameters.repetitions))
    ESS = np.zeros((1, parameters.Time, parameters.repetitions))
    for i_simulation in range(parameters.repetitions):
    #for i_simulation in range(32):
        if propagation_method == 'true_sisson':
            try:
                simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(1)+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ) )
            except:
                simulation = pickle.load( open( parameters.filename+'_'+str(i_simulation)+"_"+sample_method+str(1)+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ) )
        elif propagation_method == 'AIS':
            try:
                simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(2)+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb"))
            except: 
                simulation = pickle.load( open( parameters.filename+'_'+str(i_simulation)+"_"+sample_method+str(2)+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb"))
        else:
            try:
                simulation = pickle.load( open( parameters.filename+str(i_simulation)+"_"+sample_method+str(parameters.kwargs["dim_auxiliary_var"])+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ))
            except:
                simulation = pickle.load( open( parameters.filename+'_'+str(i_simulation)+"_"+sample_method+str(parameters.kwargs["dim_auxiliary_var"])+"_"+str(propagation_method)+"_"+str(particles)+"_simulation_abc_epsilon_"+str(parameters.epsilon_target)+".p", "rb" ))

        selector = simulation["means_particles"].shape[1]
        #pdb.set_trace()
        #if propagation_method == 'Del_Moral':
        #    selector = selector - 1
        #pdb.set_trace()
        means[:, :selector, i_simulation] = simulation["means_particles"][:, :selector]
        # calculate the L1 distances
        try:
            for t_simulation in range(selector):
                l1_distances[:, t_simulation, i_simulation] = simulation_parameters_model.functions_model.l1_distance(simulation["particles"][:, :, t_simulation])
        except: pdb.set_trace()
        vars[:, :selector, i_simulation] = np.atleast_3d(simulation["var_particles"])[0,:, :selector]
        final_ESS[:,i_simulation] = simulation["ESS"][selector-1]
        final_epsilon[:,i_simulation] = simulation["epsilon"][selector-1]
        epsilons[:,:len(simulation["epsilon"]),i_simulation] = simulation["epsilon"]
        #pdb.set_trace()
        ESS[:,:len(simulation["ESS"]),i_simulation] = simulation["ESS"]
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

    if propagation_method=='Del_Moral':
        selector = simulation["means_particles"].shape[1] -1
    means_all = means[:, :selector, :]
    l1_distances = l1_distances[:, :selector, :]
    vars_all = vars[:, :selector, :]
    epsilons = epsilons[:,:len(simulation["epsilon"]),:]
    ESS = ESS[:,:len(simulation["ESS"]),:]
    var_all = means_all.var(axis=2)
    vars_vars = vars_all.var(axis=2)
    vars_means = vars_all.mean(axis=2)
    means_last = np.nanmean(means_all[:,-1,:], axis=1)
    vars_vars_last = vars_vars[:,-1]
    vars_means_last = vars_means[:,-1]
    #means_var = np.nanvar(final_means, axis=1)
    means_var_last = (means_all[:,-1,:]**2).mean(axis=1) # use this for the MSE
    ESS_mean = final_ESS.mean()
    epsilon_mean = final_epsilon.mean()
    time_mean = final_simulation_time.mean()
    number_simulations_mean = final_number_simulations.mean()
    number_simulations = number_simulations[0,:, :selector]
    #pdb.set_trace()
    return [means_last, means_var_last, vars_means_last, vars_vars_last, ESS_mean, epsilon_mean, time_mean, number_simulations_mean], [means_all, epsilons, means_all.var(axis=2), number_simulations, vars_vars, vars_all.mean(axis=2), vars_all, l1_distances, ESS]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#pdb.set_trace()
#sisson_simulation_results = f_summary_stats(sisson_simulation_parameters_mixture_model, sample_method = "MC", particles=100)
if False:
    var_different_methods = np.zeros((5,len(simulation_parameters_model.kwargs['N_particles_list']), simulation_parameters_model.kwargs['Time']))
    counter = 0
    for N_particles in simulation_parameters_model.kwargs['N_particles_list']:
        QMC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "QMC", particles=N_particles, propagation_method = 'AIS')
        #pdb.set_trace()
        MC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'AIS')
        #RQMC_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "RQMC", particles=N_particles, propagation_method = 'AIS')
        del_moral_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'Del_Moral')
        #true_sisson_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "MC", particles=N_particles, propagation_method = 'true_sisson')
        #nonparametric_simulation_results = f_summary_stats(simulation_parameters_model, sample_method = "QMC", particles=N_particles, propagation_method = 'nonparametric')
        #print sisson_simulation_results[0]
        print simulation_parameters_model.filename
        print N_particles
        print MC_simulation_results[0]
        print QMC_simulation_results[0]
        #print RQMC_simulation_results[0]
        print del_moral_simulation_results[0]
        #print nonparametric_simulation_results[0]
        #print true_sisson_simulation_results[0]
        #pdb.set_trace()

def function_flatten_results(_results, dim, method="other"):
    _means_inter = _results[1][0][dim, :, :].flatten()
    if method == "Del_Moral":
        _epsilons_inter = _results[1][1][:,:-1,:].flatten()
    else:
        _epsilons_inter = _results[1][1].flatten()
    _means_inter = _means_inter[_epsilons_inter > 0.]
    _epsilons_inter = _epsilons_inter[_epsilons_inter > 0.]
    return _means_inter, _epsilons_inter


def plot_no_double_epsilon(results, label):
    if label == 'Del Moral':
        #pdb.set_trace()
        plt.plot(results[1][1][0,:-1,0], (results[1][2][0,:]*results[1][3][:,:].mean(axis=0))[:], label=label, linewidth=3)
    elif label == 'Sisson':
        plt.plot(results[1][1][0,:-1,0], (results[1][2][0,:]*results[1][3].mean(axis=0))[:-1], label=label, linewidth=3)
    else:
        epsilon_list = results[1][1][0,:,0]
        epsilon_selector = epsilon_list[1:]<epsilon_list[:-1]
        #pdb.set_trace()
        #var_list = results[1][2][0,:] #(results[1][2][0,:]*results[1][3].mean(axis=0))[:]
        plt.plot(results[1][1][0,:,0], (results[1][2][0,:]*results[1][3].mean(axis=0))[:], label=label, linewidth=3)
        #plt.plot(epsilon_list[epsilon_selector], var_list[epsilon_selector], label=label)

def plot_no_double_epsilon_variance(results, label, true_variance=1):
    #pdb.set_trace()
    vars_all = results[1][6][0,:]
    mse_vars_all = ((vars_all-true_variance)**2).mean(axis=1)
    if label == 'Del Moral':
        #pdb.set_trace()
        plt.plot(results[1][1][0,:-1,0], (mse_vars_all*results[1][3][:,:].mean(axis=0))[:], label=label, linewidth=3)
    elif label == 'Sisson':
        pdb.set_trace()
        plt.plot(results[1][1][0,:-1,0], (mse_vars_all*results[1][3].mean(axis=0))[:-1], label=label, linewidth=3)
    else:
        #epsilon_list = results[1][1][0,:,0]
        #epsilon_selector = epsilon_list[1:]<epsilon_list[:-1]
        #pdb.set_trace()
        #var_list = results[1][2][0,:] #(results[1][2][0,:]*results[1][3].mean(axis=0))[:]
        plt.plot(results[1][1][0,:,0], (mse_vars_all*results[1][3].mean(axis=0))[:], label=label, linewidth=3)
        #plt.plot(epsilon_list[epsilon_selector], var_list[epsilon_selector], label=label)

def plot_no_double_epsilon_l1_distance(results, label):
    #pdb.set_trace()
    if label == 'Del Moral':
        #pdb.set_trace()
        plt.plot(results[1][1][0,:-1,0], (results[1][-2].mean(axis=2))[0,:], label=label, linewidth=3)
    elif label == 'Sisson':
        plt.plot(results[1][1][0,:-1,0], (results[1][-2].mean(axis=2))[0,:-1], label=label, linewidth=3)
    else:
        epsilon_list = results[1][1][0,:,0]
        epsilon_selector = epsilon_list[1:]<epsilon_list[:-1]
        pdb.set_trace()
        plt.plot(results[1][1][0,:,0], (results[1][-2].mean(axis=2))[0,:], label=label, linewidth=3)
        #plt.plot(epsilon_list[epsilon_selector], var_list[epsilon_selector], label=label)

def plot_no_double_epsilon_ESS(results, label):
    #pdb.set_trace()
    if label == 'Del Moral':
        plt.plot(results[1][1][0,:-1,0], (results[1][-1].mean(axis=2))[0,:-1], label=label, linewidth=3)
    elif label == 'Sisson':
        #pdb.set_trace()
        plt.plot(results[1][1][0,:-1,0], (results[1][-1].mean(axis=2))[0,:-1], label=label, linewidth=3)
    else:
        epsilon_list = results[1][1][0,:,0]
        epsilon_selector = epsilon_list[1:]<epsilon_list[:-1]
        #pdb.set_trace()
        plt.plot(results[1][1][0,:,0], (results[1][-1].mean(axis=2))[0,:], label=label, linewidth=3)



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
        pickle.dump( results_summary_to_save, open( str(N_particles)+"_results_summary_to_save.p", "wb" ) )
        pdb.set_trace()

        print simulation_parameters_model.filename
        print N_particles
        print MC_results[0]
        print QMC_results[0]
        print RQMC_results[0]
        print Del_Moral_results[0]
        print Sisson_results[0]

        sns.set_style("darkgrid")
        #sns.tsplot(time=MC_epsilons_inter, data=MC_means_inter, color='blue')
        #sns.tsplot(time=RQMC_epsilons_inter, data=RQMC_means_inter, color='green')
        #sns.tsplot(time=Del_Moral_epsilons_inter, data=Del_Moral_means_inter, color='red')
        #plt.subplot(1,1,1)
        #pdb.set_trace()
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
