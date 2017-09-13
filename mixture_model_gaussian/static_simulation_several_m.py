#from future import __division__
import ipdb as pdb
import cPickle as pickle
import numpy as np

import os
root_path = "/home/alex/python_programming"
import sys
import copy

from scipy.stats import norm

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(root_path+"/ABC/oo_sampler/functions")

sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")
sys.path.append(root_path+"/ABC")
from functions_static_simulation import *









if __name__ == '__main__':
    
    import functions_mixture_model_3 as functions_model
    simulator = functions_model.simulator_vectorized
    delta = functions_model.delta_vectorized
    theta_sampler_mc = functions_model.theta_sampler_mc
    theta_sampler_rqmc = functions_model.theta_sampler_rqmc
    theta_sampler_qmc = functions_model.theta_sampler_qmc
    theta_sampler_list = [theta_sampler_mc, theta_sampler_qmc, theta_sampler_rqmc]
    sampler_type_list = ['MC', 'QMC', 'RQMC']

    dim = 1
    N_simulations = 10**5
    M_repetitions = 75
    m_intra = 50

    y_star = functions_model.f_y_star(dim)
    instance_compare_samplers = compare_sampling_methods(M_repetitions, simulator, delta, dim, N_simulations, y_star, m_intra=m_intra)

    threshold = 1.

    #instance_compare_samplers.generate_samples(theta_sampler_qmc, "QMC")
    #accepted_qmc = (instance_compare_samplers.reference_table_distances_qmc<threshold)[0,:,:,:]

    instance_compare_samplers.generate_samples(theta_sampler_rqmc, "RQMC")
    accepted_rqmc = (instance_compare_samplers.reference_table_distances_rqmc<threshold)[0,:,:,:]

    #instance_compare_samplers.generate_samples(theta_sampler_mc, "MC")
    #accepted_mc = (instance_compare_samplers.reference_table_distances_mc<threshold)[0,:,:,:]

    lower_bound = -10
    upper_bound =  10

    def acceptance_probability(lower_bound, upper_bound, threshold):
        a = (threshold + upper_bound) *norm.cdf(threshold + upper_bound)+norm.pdf(threshold + upper_bound)
        b = (threshold + lower_bound) *norm.cdf(threshold + lower_bound)+norm.pdf(threshold + lower_bound)
        c = (-threshold + upper_bound) *norm.cdf(-threshold + upper_bound)+norm.pdf(-threshold + upper_bound)
        d = (-threshold + lower_bound) *norm.cdf(-threshold + lower_bound)+norm.pdf(-threshold + lower_bound)
        return (a - b - (c - d))/(upper_bound - lower_bound)
    print acceptance_probability(lower_bound, upper_bound, threshold)
    #print accepted_mc.mean()
    print accepted_rqmc.mean()

    def acceptance_probability_fixed_theta(theta, threshold):
        return norm.cdf(threshold + theta)-norm.cdf(theta - threshold)



    def extract_variance(m_intra, M_repetitions, accepted):
        var_mean_during = np.zeros((m_intra-1, M_repetitions))
        mean_var_during = np.zeros(((m_intra-1, M_repetitions)))
        var_after = np.zeros(m_intra-1)
        m_list = np.arange(2,m_intra+1)
        """ accepted[1, N_particles, m_intra, M_repetitions]"""
        for m in m_list:
            accepted_inter = np.copy(accepted[:,:m,:])
            #pdb.set_trace()

            conditional_variance = accepted_inter.mean(axis=1)*(1.-accepted_inter.mean(axis=1))
            mean_conditional_variance = m*conditional_variance.mean(axis=0)/(N_simulations*(m-1))

            conditional_mean = accepted_inter.mean(axis=0)
            variance_conditional_mean = conditional_mean.mean(axis=0)*(1-conditional_mean.mean(axis=0))

            var_mean_during[m-2, :] = variance_conditional_mean*m#np.var(np.mean(accepted_inter, axis=1), axis=0)*(m**2)/((m-1.)*N_simulations)
            mean_var_during[m-2, :] = mean_conditional_variance
            #pdb.set_trace()
            var_after[m-2] = accepted_inter.mean(axis=0).mean(axis=0).var()*m
            del accepted_inter
        return mean_var_during, var_mean_during, var_after


    true_variance = acceptance_probability(lower_bound, upper_bound, threshold)*(1.-acceptance_probability(lower_bound, upper_bound, threshold))
    thetas_integration = np.linspace(-10,10, num=100000)

    var_mean_numeric = (acceptance_probability_fixed_theta(thetas_integration, threshold)**2 - acceptance_probability_fixed_theta(thetas_integration, threshold).mean()**2).mean()
    mean_var_numeric = np.mean(acceptance_probability_fixed_theta(thetas_integration, threshold)*(1.-acceptance_probability_fixed_theta(thetas_integration, threshold)))

    #mean_var_during_mc, var_mean_during_mc, var_after_mc = extract_variance(m_intra, M_repetitions, accepted_mc)
    #mean_var_during_qmc, var_mean_during_qmc, var_after_qmc = extract_variance(m_intra, M_repetitions, accepted_qmc)
    mean_var_during_rqmc, var_mean_during_rqmc, var_after_rqmc = extract_variance(m_intra, M_repetitions, accepted_rqmc)
    #pdb.set_trace()
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid", {'axes.grid' : False})
    #plt.boxplot(list_during_boxplot)
    #plt.plot(mean_var_during_mc.mean(axis=1), label="Mean Var MC", linewidth=3, linestyle='dotted')
    #plt.plot(var_mean_during_mc.mean(axis=1), label="Var Mean MC", linewidth=3, linestyle='dashed')
    #plt.plot(var_after_mc, label='Var standard MC', linewidth=3)

    def plot_results(type_plot, mean_var_during, var_mean_during, var_after, m_intra=m_intra):
        #pdb.set_trace()
        #plt.plot(mean_var_during.mean(axis=1), label="Variance estimator intra "+type_plot, linewidth=3, linestyle='dashed')
        plt.plot(mean_var_during[:,0], label="Variance estimator intra "+type_plot, linewidth=3, linestyle='dashed')
        #plt.plot(mean_var_during[:,0], label="Mean Var "+type_plot, linewidth=3, linestyle='dashed')

        #plt.plot(var_mean_during.mean(axis=1), label="Var Mean "+type_plot, linewidth=3, linestyle='dashed')
        plt.plot(var_after, label='Variance several runs '+type_plot, linewidth=3)
        #ones = np.ones(var_after_qmc.shape)
        #ones = np.arange(2,m_intra+1)
        #plt.plot(ones*true_variance, label='true variance', linewidth=2)
        #plt.plot(ones*var_mean_numeric, label='true var mean', linewidth=2, linestyle='dotted')
        #plt.plot(ones*mean_var_numeric, label='true mean var', linewidth=2, linestyle='dotted')
        #plt.plot((mean_var_during.mean(axis=1)+var_mean_during.mean(axis=1)), label='sum', linewidth=3)

        plt.legend(fontsize=14)
        plt.yscale('log')#; plt.xscale('log')
        plt.ylabel('Variance times M', fontsize=14); plt.xlabel('M', fontsize=14)
        plt.savefig(type_plot+'variance_during_after_several_m.png')
        
    #plt.figure()
    #plot_results("MC", mean_var_during_mc, var_mean_during_mc, var_after_mc)
    #plt.figure()
    #plot_results("QMC", mean_var_during_qmc, var_mean_during_qmc, var_after_qmc)
    plt.figure()
    plot_results("RQMC", mean_var_during_rqmc, var_mean_during_rqmc, var_after_rqmc)
    plt.show()
    pdb.set_trace()




