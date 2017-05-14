# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:26:07 2017

@author: alex
Simulation started on friday 15:45 20.1.2017
"""
from functools import partial
import numpy as np
import sys
import ipdb as pdb


sys.path.append("/home/alex/python_programming/ABC/oo_sampler/class_method_smc")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/alpha_stable_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/lotka_volterra_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")


path = "/media/alex/Transcend/ABC_results_storage/simulation_results_12_5_17"
import gaussian_densities_etc
import functions_tuberculosis_model as functions_model
#import functions_mixture_model as functions_model
#import functions_lotka_volterra_model as functions_model
from class_smc import smc_sampler
import functions_propagate_reweight_resample

Time = 600
dim_particles = 2
repetitions = 4
N_particles = 10
#filename = functions_model.model_string+'_dim_'+str(dim_particles)+'_adaptive_M_autochoose_eps_gaussian_kernel'
filename = functions_model.model_string+'_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3'



#import plot_bivariate_scatter_hist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
model_description = functions_model.model_string

Time = 100
dim_auxiliary_var = 2
augment_M = False
M_incrementer = 2
target_ESS_ratio_reweighter = 0.5
target_ESS_ratio_resampler = 0.5
epsilon_target = 1000
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
y_star = functions_model.f_y_star(dim_particles)


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


simulator_mm = functions_propagate_reweight_resample.simulator_sampler(functions_model.simulator,
                                    y_star,
                                    functions_model.delta,
                                    functions_model.exclude_theta,
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

test_sampler.setInitiationFunction(functions_model.theta_sampler_rqmc)
length_quantiles = 20
quantiles = np.linspace(0.1, 0.005, num=length_quantiles)


sampler_list = ['mc', 'qmc', 'rqmc']
repetitions = 8
#pdb.set_trace()
from matplotlib import pyplot as plt
#plt.figure()
import pickle
N_particles = 10**2
dim_particles = 2
import os
#os.chdir(path)

def parallel_sampler(iter, test_sampler, N_particles, quantiles):
    """
    the function that will be parallelized 
    """
    test_sampler.accept_reject_sampler(N_particles)
    particles = test_sampler.particles_AR_posterior
    distances = test_sampler.auxialiary_particles_accept_reject
    results = {}
    results['particles'] = particles
    results['distances'] = distances
    means_list = []
    vars_list = []
    for j_quantile in range(length_quantiles):
        posterior = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), percentile=quantiles[j_quantile])
        #pdb.set_trace()
        means_list.append(posterior.mean(axis=1))
        vars_list.append(np.cov(posterior))
    results['means'] = means_list
    results['vars'] = vars_list
    results['quantiles'] = quantiles
    return(results)


from multiprocessing import Process, Pipe
from itertools import izip
import multiprocessing

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]


if True:
    results_list = {}
    for sampler in sampler_list:
        test_sampler.dim_particles = dim_particles
        del test_sampler.f_initiate_particles
        #pdb.set_trace()
        results_list_intra_sampler = []
        if sampler == 'mc':
            test_sampler.setInitiationFunction(functions_model.theta_sampler_mc)
        elif sampler == 'qmc':
            test_sampler.setInitiationFunction(functions_model.theta_sampler_qmc)
        elif sampler == 'rqmc':
            test_sampler.setInitiationFunction(functions_model.theta_sampler_rqmc)
        else: raise ValueError('error in sampler!')
        array_results = np.zeros((2, length_quantiles, repetitions, dim_particles))
        array_results_posterior_distance_sampler = []
        partial_parallel_sampler = partial(parallel_sampler, test_sampler=test_sampler, N_particles=N_particles, quantiles=quantiles)
        NUM_CORES = multiprocessing.cpu_count()
        #pdb.set_trace()
        list_repetitions = range(repetitions)
        chunks = [list_repetitions[i:i + NUM_CORES] for i in range(0, len(list_repetitions), NUM_CORES)] 
        for chunk in chunks:
            F_results = parmap(partial_parallel_sampler, chunks)
            results_list_intra_sampler.append(F_results)
        # unlists the results and puts them in one list
        results_list[sampler] = [item for sublist in results_list_intra_sampler for item in sublist]
        #pdb.set_trace()
    pickle.dump(results_list, open(functions_model.model_string+"means_static_simulation_gaussian_mixuture_dim"+str(dim_particles)+".p", "wb") )
    #pickle.dump(array_results_list_posterior_distance, open(functions_model.model_string+"particles_distances_static_simulation_gaussian_mixuture_dim"+str(dim_particles)+".p", "wb") )
    #pdb.set_trace()

if False:

    plt.title("Variance of the variance estimator, dimension "+str(dim_particles), fontsize=18)
    plt.plot(quantiles, array_results_list[0].var(axis=2).sum(axis=2)[1,:], label="MC", linewidth=3)
    plt.plot(quantiles, array_results_list[1].var(axis=2).sum(axis=2)[1,:], label="QMC", linewidth=3)
    plt.plot(quantiles, array_results_list[2].var(axis=2).sum(axis=2)[1,:], label="RQMC", linewidth=3)
    plt.xlabel('Quantile of distance', fontsize=14); plt.ylabel('Variance of estimator', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.savefig(functions_model.model_string+"variance_of_variance_estimator_dim"+str(dim_particles)+".png")
    plt.clf()


    plt.title("Variance of the mean estimator, dimension "+str(dim_particles), fontsize=18)
    plt.plot(quantiles, array_results_list[0].var(axis=2).sum(axis=2)[0,:], label="MC", linewidth=3)
    plt.plot(quantiles, array_results_list[1].var(axis=2).sum(axis=2)[0,:], label="QMC", linewidth=3)
    plt.plot(quantiles, array_results_list[2].var(axis=2).sum(axis=2)[0,:], label="RQMC", linewidth=3)
    plt.xlabel('Quantile of distance', fontsize=14); plt.ylabel('Variance of estimator', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.savefig(functions_model.model_string+"variance_of_mean_estimator_dim"+str(dim_particles)+".png")
    plt.clf()

# posterior qmc
if True:
    test_sampler.setInitiationFunction(functions_model.theta_sampler_qmc)
    test_sampler.accept_reject_sampler(N_particles)
    posterior_qmc = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), percentile=0.001)

    test_sampler.setInitiationFunction(functions_model.theta_sampler_mc)
    test_sampler.accept_reject_sampler(N_particles)
    posterior_mc = test_sampler.f_accept_reject_precalculated_particles(test_sampler.particles_AR_posterior, test_sampler.auxialiary_particles_accept_reject.flatten(), percentile=0.001)

    
    pdb.set_trace()
    import pandas as pd
    #from pandas.plotting import scatter_matrix
    df_mc = pd.DataFrame(posterior_mc.transpose(), columns=['1', '2', '3'])
    df_mc['type'] = 'mc'
    df_qmc = pd.DataFrame(posterior_qmc.transpose(), columns=['1', '2', '3'])
    df_qmc['type'] = 'qmc'
    frames = [df_mc, df_qmc]
    result = pd.concat(frames)
    sns.pairplot(df_mc); plt.show()
    sns.pairplot(df_qmc); plt.show()
    sns.pairplot(result, hue='type', palette="husl", plot_kws={"s":40, "alpha":.5,'lw':1, 'edgecolor':'k'}); plt.show()

    pdb.set_trace()
    '''g = sns.PairGrid(df_qmc)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)

    sns.kdeplot(posterior_mc[0,:], posterior_mc[1,:], label="MC posterior", cmap="Reds", shade=True, shade_lowest=False)
    sns.kdeplot(posterior_qmc[0,:], posterior_qmc[1,:], label="QMC posterior", cmap="Blues", shade=True, shade_lowest=False)
    sns.jointplot(x=posterior_mc[0,:], y=posterior_mc[1,:], kind="kde")

    plt.legend(fontsize=14)
    plt.savefig("posterior_distribution_mixed_gaussian.png")
    plt.show()
    '''
    #import yappi
    #yappi.start()
    #test_sampler.iterate_smc(resample=resample, save=save, modified_sampling=propagation_mechanism)
    #yappi.get_func_stats().print_all()

