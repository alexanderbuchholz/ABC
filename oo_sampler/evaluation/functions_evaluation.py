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
import os
root_path = "/home/alex/python_programming"
import sys
sys.path.append(root_path+"/ABC/oo_sampler/simulation")
sys.path.append(root_path+"/ABC/oo_sampler/functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")
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
                l1_distances[:, t_simulation, i_simulation] = parameters.functions_model.l1_distance(simulation["particles"][:, :, t_simulation])
        except: pdb.set_trace()
        vars[:, :selector, i_simulation] = np.atleast_3d(simulation["var_particles"])[0, :, :selector]
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

def resample_for_plotting(particles, weights):
    particles_resampled = np.zeros(particles.shape)
    N_particles = particles.shape[1]
    import resampling
    ancestors = resampling.stratified_resample(weights.squeeze())
    particles_resampled = particles[:, ancestors]
    gaussian_densities_etc.break_if_nan(particles_resampled)
    return particles_resampled


def f_accept_reject_precalculated_particles(precalculated_particles, precalculated_auxialiary_particles, epsilon_target_accept_reject):
    accept_reject_selector = precalculated_auxialiary_particles < epsilon_target_accept_reject
    return precalculated_particles[:, accept_reject_selector]