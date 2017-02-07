# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:44:54 2016

@author: alex
"""
from __future__ import print_function
import numpy as np
import numpy.random as nr
import pickle
import ipdb as pdb
import time

import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
import gaussian_densities_etc
import functions_propagate_reweight_resample


class smc_sampler(object):
    """
        particles have the following size :
            (dim_particles, N_particles, T_time)

    """
    def __init__(self, N_particles, dim_particles, Time, ESS_treshold_resample=None, ESS_treshold_incrementer = None, dim_auxiliary_var = 0, augment_M=False, epsilon_target=0.05, contracting_AIS=False, M_incrementer=5, M_increase_until_acceptance=True, M_target_multiple_N=1.):
        """
            set the data structures of the class
            set the random generator that will drive the stochastic propagation
        """
        self.T = Time
        self.N_particles = N_particles
        self.dim_particles = dim_particles
        if ESS_treshold_resample == None:
            self.ESS_treshold_resample = N_particles*0.3
        else:
            self.ESS_treshold_resample = ESS_treshold_resample
        if ESS_treshold_incrementer == None:
            self.ESS_treshold_incrementer = N_particles*0.3
        else:
            self.ESS_treshold_incrementer = ESS_treshold_incrementer
        # defining the interior data structure
        self.particles = np.zeros(shape=(self.dim_particles, self.N_particles, self.T+1))
        self.particles_before_resampling = np.zeros(shape=(self.dim_particles, self.N_particles, self.T+1))
        self.particles_preweights = np.zeros(shape=(1, self.N_particles, self.T+1))
        #self.particles_resampled = np.zeros(shape=(self.dim_particles, self.N_particles, self.T))
        self.weights = np.ones(shape=(1, self.N_particles, self.T+1))*1./self.N_particles
        self.sampling_counter = 0.
        self.information_components = []
        self.augment_M = augment_M
        self.epsilon_target = epsilon_target
        self.contracting_AIS = contracting_AIS
        self.M_incrementer = M_incrementer
        self.M_increase_until_acceptance = M_increase_until_acceptance
        self.M_target_multiple_N = M_target_multiple_N
        if dim_auxiliary_var > 0:
            # the dim of the auxiliary var is the number of additional simulations
            self.dim_auxiliary_var = dim_auxiliary_var
            self.auxialiary_particles_list = []
            self.auxialiary_particles = np.zeros(shape=(self.dim_auxiliary_var,
                                                        self.N_particles,
                                                        self.T+1))
            self.auxialiary_weights = np.ones(shape=(1, self.N_particles, self.T+1))*1./self.N_particles

        else:
            self.dim_auxiliary_var = 0
        self.epsilon = None
        self.ESS = np.ones(self.T+1)
        self.ESS[0] = self.N_particles
        self.ESS_before_reweighting = np.ones(self.T+1)
        self.mean_particles = np.zeros((self.dim_particles, self.T+1))
        self.var_particles = np.zeros((self.dim_particles, self.dim_particles, self.T+1))

    def setParameters(self, parameters):
        self.parameters = parameters

    def setInitiationFunction(self, function, parameters={}):
        self.f_initiate_particles = function
        self.parameters_Initiation = parameters

    def setPropagateFunction(self, function, parameters={}):
        self.f_propagate_particles = function
        self.parameters_Propagate = parameters

    def setReweightFunction(self, function, parameters={}):
        self.f_weight_particles = function
        self.parameters_Reweight = parameters

    def setResampleFunction(self, function, parameters={}):
        self.f_resample_particles = function
        self.parameters_Resample = parameters

    def setRandomGenerator(self, function, parameters={}):
        self.random_sequence = function
        self.parameters_RandomGenerator = parameters

    def setAuxialiarySampler(self, function, parameters={}):
        self.class_auxialiary_sampler = function
        self.parameters_AuxialiarySampler = parameters
    def setEpsilonSchedule(self, epsilon):
        self.epsilon = epsilon # override epsilon
        if len(epsilon) != self.T:
            
            raise ValueError('Epsilon schedule does not correspond to time steps !')

    def __initialize_sampler(self, *args, **kwargs):
        """
            sample particles to start with

        """
        i = 0
        dim = self.dim_particles
        n = self.N_particles

        kwargs = self.parameters_Initiation
        self.particles[:, :, 0] = self.f_initiate_particles(i, dim, n, kwargs)
        self.particles_before_resampling[:, :, 0] = self.particles[:, :, 0]
        self.particles_preweights[:, :, 0] = 1./self.N_particles

        # sampling of auxialiray particles
        if self.dim_auxiliary_var > 0:
            kwargs = self.parameters_AuxialiarySampler
            self.auxialiary_particles_list.append(self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:,:,0], **kwargs))
            #pdb.set_trace()
            #self.auxialiary_particles[:,:,0] = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:,:,0], **kwargs)

    def accept_reject_sampler(self, N_particles_AR, epsilon_target_accept_reject):
        """
            sample particles to start with

        """
        i = 0
        dim = self.dim_particles
        n = N_particles_AR

        kwargs = self.parameters_Initiation
        self.particles_AR_initial = self.f_initiate_particles(i, dim, n, kwargs)

        # sampling of auxialiray particles
        if self.dim_auxiliary_var > 0:
            kwargs = self.parameters_AuxialiarySampler
            M_simulator_inter = self.class_auxialiary_sampler.M_simulator
            self.class_auxialiary_sampler.M_simulator = 1
            self.auxialiary_particles_accept_reject = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles_AR_initial, **kwargs)
            self.accept_reject_selector = self.auxialiary_particles_accept_reject < epsilon_target_accept_reject
            #pdb.set_trace()
            self.particles_AR_posterior = self.particles_AR_initial[self.accept_reject_selector]
            self.class_auxialiary_sampler.M_simulator = M_simulator_inter 
            #self.auxialiary_particles[:,:,0] = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:,:,0], **kwargs)

    def f_accept_reject_precalculated_particles(self, precalculated_particles, precalculated_auxialiary_particles, epsilon_target_accept_reject):
        accept_reject_selector = precalculated_auxialiary_particles < epsilon_target_accept_reject
        return precalculated_particles[:, accept_reject_selector]


    def propagate_particles(self, current_t, flag_failed_ESS=False, *args, **kwargs):
        """
            move particles from t-1 to t
        """
        kwargs = self.parameters_Propagate
        if current_t == 0:
            self.__initialize_sampler()
        else:
            self.particles[:, :, current_t], self.particles_preweights[:, :, current_t], information_components = self.f_propagate_particles(self.particles[:, :, current_t-1], 
                                                                                                                    self.weights[:, :, current_t-1], 
                                                                                                                    flag_failed_ESS, 
                                                                                                                    self.information_components, 
                                                                                                                    kwargs)
            self.particles_before_resampling[:, :, current_t] = self.particles[:, :, current_t]
            self.information_components.append(information_components)
            # sampling of auxialiray particles
            if self.dim_auxiliary_var > 0:
                kwargs = self.parameters_AuxialiarySampler
                #pdb.set_trace()
                if self.M_increase_until_acceptance == True:
                    auxiliary_particles_new = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:, :, current_t], **kwargs)
                    M_simulator_inter = self.class_auxialiary_sampler.M_simulator
                    self.class_auxialiary_sampler.M_simulator = 1
                    auxiliary_particles_list_length = len(self.auxialiary_particles_list)
                    self.auxialiary_particles_list.append(auxiliary_particles_new)
                    #pdb.set_trace()
                    y_alive = np.sum((auxiliary_particles_new < self.epsilon[current_t-1]).flatten())
                    counter_M_simulations = 0
                    while y_alive<self.N_particles*self.M_target_multiple_N:
                        print('Successfull simulations y = %s of in total M*target= %s, number of tries M %s' % (y_alive, self.N_particles*self.M_target_multiple_N, counter_M_simulations), end='\r')
                        auxiliary_particles_inter = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:, : ,current_t], **kwargs)
                        auxiliary_particles_new = np.vstack((auxiliary_particles_new, auxiliary_particles_inter))
                        y_alive = np.sum((auxiliary_particles_new < self.epsilon[current_t-1]).flatten())
                        self.auxialiary_particles_list[auxiliary_particles_list_length] = auxiliary_particles_new
                        counter_M_simulations += 1
                    print('\n')
                    #pdb.set_trace()
                    self.class_auxialiary_sampler.M_simulator = M_simulator_inter+counter_M_simulations
                else: 
                    auxiliary_particles_new = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:, :, current_t], **kwargs)
                    self.auxialiary_particles_list.append(auxiliary_particles_new)
                #import matplotlib.pyplot as plt
                #X =  self.auxialiary_particles_list[-1].mean(axis=0)
                #print X.mean()
                #plt.hist(X)
                #plt.show()

                #self.auxialiary_particles[:,:,current_t] = self.class_auxialiary_sampler.f_auxialiary_sampler(self.particles[:,:,current_t], **kwargs)
        gaussian_densities_etc.break_if_nan(self.particles[:,:,current_t])
        gaussian_densities_etc.break_if_nan(self.weights[:,:,current_t])

    def propagate_particles_sisson(self, current_t, *args, **kwargs):
        """
        function that implements the propagation of sisson et al.
        """
        #kwargs = self.parameters_Propagate
        if self.class_auxialiary_sampler.M_simulator > 1.:
            raise ValueError('simulator of y needs to return only one value!')
        if current_t==0:
            self.__initialize_sampler() # TODO: not correct, change this !
            self.M_list.append(self.N_particles)
        else:
            particles_var = np.atleast_2d(2*np.cov(self.particles[:,:,current_t-1], aweights=np.squeeze(self.weights[:,:,current_t-1])))
            #particles_next = np.zeros(self.particles[:,:,current_t-1].shape)
            counter_M = 0.
            for index_particle in xrange(self.N_particles):
                # TODO: chose ancestor
                u = nr.uniform()
                ancestor = gaussian_densities_etc.weighted_choice(np.squeeze(self.weights[:,:,current_t-1]),u)
                center = self.particles[:, ancestor, current_t-1]
                dist = 100000000000.
                #pdb.set_trace()
                
                while dist> self.epsilon[current_t]: # this amounts to a uniform acceptance
                    #pdb.set_trace()
                    proposal = gaussian_densities_etc.gaussian_standard(center, particles_var)[:,np.newaxis]
                    dist = self.class_auxialiary_sampler.f_auxialiary_sampler(proposal)
                    self.sampling_counter += 1.
                    counter_M += 1.
                self.particles[:,index_particle, current_t] = proposal.squeeze()
                self.auxialiary_particles[:,index_particle, current_t] = dist

            self.M_list.append(counter_M)
            for i_particle in xrange(self.N_particles):
                # TODO: calc preweights
                #pdb.set_trace()
                self.particles_preweights[:,i_particle,current_t] = np.array([self.weights[:,i_former_particle,current_t-1]*gaussian_densities_etc.gaussian_density(self.particles[:,i_particle,current_t], self.particles[:,i_former_particle,current_t], particles_var) for i_former_particle in range(self.N_particles)]).sum()
            self.auxialiary_particles_list.append(self.auxialiary_particles[:,:, current_t])
            self.particles_before_resampling[:, :, current_t] = self.particles[:, :, current_t]



    def reweight_particles(self, current_t, *args, **kwargs):
        """
            1. weighting with auxiliary particles
            2. weighting without auxiliary particles
        """
        # TODO: continue here
        #pdb.set_trace()
        kwargs = self.parameters_Reweight
        if self.dim_auxiliary_var>0:
            #kwargs = self.parameters_AuxialiarySampler
            #pdb.set_trace()
            if current_t == 0:
                previous_ESS = self.N_particles
            else:
                previous_ESS = self.ESS[current_t-1]
            weights, particles_mean, particles_var, ESS, epsilon_current, ESS_before_reweighting = self.f_weight_particles(self.particles[:,:,current_t],
                                                                                                    self.particles_preweights[:,:,current_t],
                                                                                                    current_t,
                                                                                                    #aux_particles=self.auxialiary_particles[:,:,current_t],
                                                                                                    aux_particles=self.auxialiary_particles_list[current_t],
                                                                                                    weights_before = self.weights[:,:,current_t-1],
                                                                                                    epsilon=self.epsilon, previous_ESS=previous_ESS, **kwargs)
        else:
            assert False # this is not implemented !
        self.weights[:,:,current_t]= weights
        self.mean_particles[:,current_t] = particles_mean
        self.var_particles[:,:,current_t] = particles_var
        self.ESS[current_t] = ESS
        self.ESS_before_reweighting[current_t] = ESS_before_reweighting
        #pdb.set_trace()
        self.epsilon[current_t] = epsilon_current
        gaussian_densities_etc.break_if_nan(self.particles)
        gaussian_densities_etc.break_if_nan(self.weights)

    def resample_particles(self, current_t, *args, **kwargs):
        """
        routine for resampling of the particles
        """
        kwargs = self.parameters_Resample
        #pdb.set_trace()
        self.particles[:,:,current_t], self.weights[:,:,current_t] = self.f_resample_particles(self.weights[:,:,current_t],
                                                                                                    self.particles[:,:,current_t],
                                                                                                    self.ESS[current_t],
                                                                                                    self.ESS_treshold_resample, **kwargs)
        #pdb.set_trace()
        self.ESS[current_t] = 1./np.sum(self.weights[:,:,current_t]**2)
        gaussian_densities_etc.break_if_nan(self.particles)
        gaussian_densities_etc.break_if_nan(self.weights)

    def metropolis_hasting_accept_reject(self, current_t, *args, **kwargs):
        #pdb.set_trace()
        aux_particles_new = self.auxialiary_particles_list[current_t]
        aux_particles_old = self.auxialiary_particles_list[current_t-1]
        accept_reject_indicator = functions_propagate_reweight_resample.accept_reject_del_moral(self.epsilon[current_t-1], aux_particles_new, aux_particles_old)
        self.n_accepted = np.sum(accept_reject_indicator)
        self.auxialiary_particles_list[current_t] = (aux_particles_new*accept_reject_indicator)+(aux_particles_old*np.logical_not(accept_reject_indicator))
        particles_new = self.particles[:,:,current_t] 
        particles_old = self.particles[:,:,current_t-1] 
        self.particles[:,:,current_t] = (particles_new*accept_reject_indicator)+(particles_old*np.logical_not(accept_reject_indicator))

    def break_routine(self, current_t):
        self.T_max = current_t+1
        self.particles = self.particles[:,:,:current_t+1]
        self.particles_before_resampling = self.particles_before_resampling[:,:,:current_t+1]
        self.weights = self.weights[:,:,:current_t+1]
        self.ESS = self.ESS[:current_t+1]
        self.ESS_before_reweighting = self.ESS_before_reweighting[:current_t+1]
        self.mean_particles = self.mean_particles[:, :current_t+1]
        self.var_particles = self.var_particles[:,:, :current_t+1]
        self.epsilon = self.epsilon[:current_t+1]

    def iterator_true_sisson(self, current_t, **kwargs):
        """
        iterator for the true sisson routine
        """
        self.propagate_particles_sisson(current_t=current_t)
        self.reweight_particles(current_t = current_t)

    def iterator_del_moral(self, current_t, **kwargs):
        """

        """
        if current_t == 0:
            self.propagate_particles(current_t)
        self.reweight_particles(current_t)
        self.resample_particles(current_t = current_t)
        #pdb.set_trace()
        self.propagate_particles(current_t+1)
        self.metropolis_hasting_accept_reject(current_t+1)

    def iterator_ais(self, current_t, resample=False, **kwargs):
        """
        iterator ais
        """
        self.propagate_particles(current_t = current_t, flag_failed_ESS=self.flag_failed_ESS)
        flag_failed_ESS = False
        self.reweight_particles(current_t = current_t)
        # choose whether to increase M
        if (self.ESS[current_t] < self.ESS_treshold_incrementer) or (self.epsilon[current_t]==self.epsilon[current_t-1]):
            if (self.augment_M):
                # first check whether increase M is true
                print("Increase M")
                self.class_auxialiary_sampler.M_simulator += self.M_incrementer
            if (not self.contracting_AIS):
                self.flag_failed_ESS = True
        self.M_list.append(self.class_auxialiary_sampler.M_simulator)
        # Resample
        if resample==True:
            self.resample_particles(current_t = current_t)
        self.sampling_counter = self.N_particles*sum(self.M_list)

    def iterator_nonparametric(self, current_t, resample=False, **kwargs):
        """
        iterator nonparametric as sisson
        """
        self.propagate_particles(current_t = current_t, flag_failed_ESS=self.flag_failed_ESS)
        flag_failed_ESS = False
        self.reweight_particles(current_t = current_t)
        # choose whether to increase M
        if (self.ESS[current_t] < self.ESS_treshold_incrementer) or (self.epsilon[current_t]==self.epsilon[current_t-1]):
            if (self.augment_M):
                # first check whether increase M is true
                print("Increase M")
                self.class_auxialiary_sampler.M_simulator += self.M_incrementer
            if (not self.contracting_AIS):
                self.flag_failed_ESS = True
        self.M_list.append(self.class_auxialiary_sampler.M_simulator)
        # Resample
        if resample==True:
            self.resample_particles(current_t = current_t)
        self.sampling_counter = self.N_particles*sum(self.M_list)
   
    def iterate_smc(self, resample=False, save=False, filename='', modified_sampling=''):
        """
        """
        # TODO: add simulation time
        self.M_list = []
        self.flag_failed_ESS = False
        self.T_max = self.T
        start_sim = time.time()
        for current_t in range(0,self.T):
            if current_t == 1:
                start = time.time()
            print("now sampling for time step %d of in total %d" %(current_t,self.T))
            ## handle true sission(self.class_auxialiary_sampler.M_simulator)
            if modified_sampling == "true_sisson":
                self.iterator_true_sisson(current_t)
            elif modified_sampling == "AIS":
                self.iterator_ais(current_t, resample=resample)
            elif modified_sampling == "nonparametric":
                self.iterator_ais(current_t, resample=resample)
            elif modified_sampling == "Del_Moral":
                self.iterator_del_moral(current_t)
            if current_t == 1:
                end = time.time()
                print ("Estimated time for the simulation in minutes %s" % ((end-start)*self.T/60.))
            if (self.epsilon_target >= self.epsilon[current_t]) or (self.epsilon[current_t]==self.epsilon[current_t-9]):
                #pdb.set_trace()
                self.break_routine(current_t)
                print("break simulation since we cannot reduce epsilon anymore or target has been reached")
                break
        end_sim = time.time()
        self.simulation_time = end_sim-start_sim
        #pdb.set_trace()
        if save == True:
            output = {'particles':self.particles,
                      'particles_before_resampling': self.particles_before_resampling,
                      'weights': self.weights,
                      'means_particles': self.mean_particles,
                      'var_particles':self.var_particles,
                      'ESS': self.ESS,
                      'ESS_before_reweighting': self.ESS_before_reweighting,
                      'epsilon':self.epsilon,
                      'M':self.dim_auxiliary_var,
                      'N':self.N_particles,
                      'simulation_time':end_sim-start_sim,
                      'sampler_typ': self.sampler_type,
                      'propagation_mechanism': self.propagation_mechanism,
                      'covar_factor': self.covar_factor,
                      'sampling_counter': self.sampling_counter,
                      'information_components': self.information_components,
                      'auxiliary_particles_list': self.auxialiary_particles_list,
                      'M_list': self.M_list,
                      'T_max': self.T_max}
            pickle.dump(output, open(filename+'_'+str(self.sampler_type)+str(self.dim_auxiliary_var)+'_'+str(self.propagation_mechanism)+'_'+str(self.N_particles)+"_simulation_abc_epsilon_"+str(self.epsilon_target)+"_"+str(self.T)+".p", "wb") )


if __name__ == '__main__':
    #import plot_bivariate_scatter_hist
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import sys
    sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
    sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model")
    sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
    sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/alpha_stable_model")
    sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
    #import functions_tuberculosis_model as functions_mixture_model
    #import functions_alpha_stable_model as functions_mixture_model
    #import functions_mixture_model_2 as functions_mixture_model
    #import functions_toggle_switch_model as functions_mixture_model
    import functions_mixture_model
    model_description = functions_mixture_model.model_string
    N_particles = 1000
    dim_particles = 4
    Time = 50
    dim_auxiliary_var = 20
    augment_M = True
    target_ESS_ratio_reweighter = 0.1
    target_ESS_ratio_resampler = 0.1
    epsilon_target = 0.01
    contracting_AIS = False
    M_increase_until_acceptance = True
    M_target_multiple_N = target_ESS_ratio_reweighter
    covar_factor = 1.
    propagation_mechanism = 'AIS'# AIS 'Del_Moral'#'nonparametric' #"true sisson" 
    sampler_type = 'MC'
    ancestor_sampling = False#"Hilbert"
    resample = True
    autochoose_eps = 'ess_based' # ''ess_based quantile_based



    model_description = model_description+'_'+sampler_type+'_'+propagation_mechanism
    save = False
    mixture_components = 10
    kernel = gaussian_densities_etc.gaussian_kernel
    move_particle =gaussian_densities_etc.gaussian_move
    y_star = functions_mixture_model.f_y_star(dim_particles)

    if autochoose_eps != '' and propagation_mechanism=='true_sisson':
        raise ValueError('if true sisson, then no autochoose_eps allowed!')

    test_sampler = smc_sampler(N_particles, 
                                dim_particles, 
                                Time, 
                                dim_auxiliary_var=dim_auxiliary_var, 
                                augment_M = augment_M, 
                                ESS_treshold_resample=N_particles*(target_ESS_ratio_resampler), 
                                ESS_treshold_incrementer = N_particles*(target_ESS_ratio_reweighter),
                                epsilon_target=epsilon_target, 
                                contracting_AIS=contracting_AIS,
                                M_increase_until_acceptance=M_increase_until_acceptance,
                                M_target_multiple_N = M_target_multiple_N)
    test_sampler.setInitiationFunction(functions_mixture_model.theta_sampler_mc)
    test_sampler.propagation_mechanism = propagation_mechanism
    test_sampler.sampler_type = sampler_type
    test_sampler.covar_factor = covar_factor
    

    simulator_mm = functions_propagate_reweight_resample.simulator_sampler(functions_mixture_model.simulator,
                                     y_star,
                                     functions_mixture_model.delta,
                                     functions_mixture_model.exclude_theta,
                                     M_simulator = dim_auxiliary_var)
    #print simulator_mm.f_auxialiary_sampler(theta)
    test_sampler.setAuxialiarySampler(simulator_mm)
    #test_sampler.initialize_sampler()


    
    propagater = functions_propagate_reweight_resample.propagater_particles(dim_particles,
                                                                            N_particles,
                                                                            move_particle,
                                                                            sampler_type=sampler_type,
                                                                            propagation_mechanism= propagation_mechanism,
                                                                            covar_factor = covar_factor,
                                                                            ancestor_sampling = ancestor_sampling,
                                                                            mixture_components = mixture_components)
    test_sampler.setPropagateFunction(propagater.f_propagate)
    #test_sampler.propagate_particles(0)
    #print test_sampler.particles

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
    #test_sampler.reweight_particles(0)
    resampler = functions_propagate_reweight_resample.resampler_particles(N_particles)
    test_sampler.setResampleFunction(resampler.f_resampling)
    if False:
        precomputed_data = functions_mixture_model.load_precomputed_data(dim_particles, functions_mixture_model.exponent)
        precalculated_particles = precomputed_data['theta_values']
        precalculated_auxialiary_particles = precomputed_data['y_diff_values']
        pdb.set_trace()
        AR_posterior_particles = test_sampler.f_accept_reject_precalculated_particles(precalculated_particles, precalculated_auxialiary_particles, epsilon_target)
        
        g = sns.distplot(AR_posterior_particles[0,:])
        plt.subplots_adjust(top=0.9)
        plt.title(('epsilon = %s \n and N = %d')% (epsilon_target, AR_posterior_particles.shape[1]))
        plt.savefig("univariate_iteration_accept_reject"+model_description+".png")
        #plt.show()
        plt.close('all')


    #pdb.set_trace()
    #import yappi
    #yappi.start()
    test_sampler.iterate_smc(resample=resample, save=save, modified_sampling=propagation_mechanism)
    #yappi.get_func_stats().print_all()
    pdb.set_trace()
    if True:
        select_component = 0
        #lim = (-0.5, 0.5)
        for i in range(test_sampler.T_max):
            if False: 
                x1_test = pd.Series(test_sampler.particles[select_component,:,i], name="$X_1$")
                x2_test = pd.Series(test_sampler.auxialiary_particles_list[i].mean(axis=0), name="$Y$")
                #g = sns.JointGrid(x=x1_test[x2_test<1000], y=x2_test[x2_test<1000], space=0)
                #g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
                #g = g.plot_marginals(sns.kdeplot, shade=True)
                g = sns.jointplot(x1_test[x2_test<1000], x2_test[x2_test<1000], kind="kde")#, xlim = lim, ylim = lim)
                plt.subplots_adjust(top=0.9)
                g.fig.suptitle(('epsilon = %s \n and N = %d')% (test_sampler.epsilon[i], test_sampler.N_particles))
                #pdb.set_trace()
                plt.savefig("bivariate_iteration_%s_%s.png"%(i, model_description))
                plt.close('all')
            
            
            #x1_new = pd.Series(test_sampler.particles[0,:,i], name = "$X_1$")
            #x2_new = pd.Series(test_sampler.particles[1,:,i], name = "$X_2$")

            #sns.jointplot(x1_new, x2_new, kind = "kde")#, xlim = lim, ylim = lim)
            g = sns.distplot(test_sampler.particles[select_component,:,i], label="model posterior")
            plt.subplots_adjust(top=0.9)
            plt.title(('epsilon = %s \n and N = %d')% (test_sampler.epsilon[i], test_sampler.N_particles))
            sns.kdeplot(AR_posterior_particles[select_component,], label="AR posterior")
            if i > 0:
                sns.kdeplot(test_sampler.particles[select_component,:,i-1], label="particles previous iteration")
            sns.kdeplot(test_sampler.particles_before_resampling[select_component,:,i], label="particles before resampling")
            plt.savefig("univariate_iteration_%s_%s_dim_%s.png"%(i, model_description, dim_particles))
            plt.show()
            plt.close()
            #plt.show()
            #plot_bivariate_scatter_hist.plot_scatter_hist(test_sampler.particles[0,:,i], test_sampler.auxialiary_particles_list[i].mean(axis=0))
            #plot_bivariate_scatter_hist.plot_scatter_hist(test_sampler.particles[0,:,i], test_sampler.particles[1,:,i])
    pdb.set_trace()
