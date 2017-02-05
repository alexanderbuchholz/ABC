# -*- coding: utf-8 -*-
# functions that do smc sampling as proposed by sisson
# this version 31.3.2016
#from functions_file import simulator, theta_sampler_rqmc, delta, theta_sampler_mc, random_sequence_mc, random_sequence_rqmc
from functions_file import *
import numpy as np
from numpy import sqrt, log, sin, cos, pi, exp
import math
import pickle
from scipy.stats import norm
import pdb
#from numba import jit, jitclass, autojit
from datetime import datetime

import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/alpha_stable_model")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/hilbert")
sys.path.append("/home/alex/python_programming/ABC/help_functions")
from functions_transform_hilbert import *
from gaussian_densities_etc import *
#from test_samplers import *

# transformation function
#@autojit()
#@profile


# define the class here
class smc_sampler_abc():
    """
    smc sampler
    """
##########################################################################################################
    def __init__(self, epsilon, N_particles, delta, y_star, simulator, random_sequence, kernel, exclude_theta, dim_theta=2, covar_factor = 3, E_treshold=None, IS=False, multiple_y=2, hilbert=False, auto_eps = False, T_max= 100, target_ESS_factor = 0.10, SMC_Sisson = False):
        self.epsilon = epsilon
        self.T = len(epsilon)
        self.y_star = y_star
        self.simulator = simulator
        self.delta = delta
        self.N_particles = N_particles
        self.random_sequence = random_sequence
        self.dim_theta = dim_theta
        self.initialized = False
        self.kernel = kernel
        self.covar_factor = covar_factor
        self.IS = IS
        self.exclude_theta = exclude_theta
        self.multiple_y = multiple_y
        self.hilbert = hilbert
        self.T_max = T_max
        self.auto_eps = auto_eps
        self.target_ESS = N_particles*target_ESS_factor
        self.SMC_Sisson = SMC_Sisson
        if E_treshold == None:
            self.E_treshold = N_particles*0.5
        else :
            self.E_treshold = E_treshold
        #self.weights = np.repeat(1./self.N_particles, self.N_particles)
##########################################################################################################

#==============================================================================
# ##########################################################################################################
#     def kernel_proposal(self, y_proposed, epsilon_t):
#         """
#         function that calculates kernel of a proposal
#         """
#         return(epsilon_t*self.kernel(self.delta(y_proposed, self.y_star) / epsilon_t))
# ##########################################################################################################
#==============================================================================

##########################################################################################################
    def f_kernel_value(self, epsilon_t, current_t, i_particle):
        """
            function that returns the kernel values
        """
        for i_multiple_y in range(self.multiple_y):
            self.kernel_values_y[i_multiple_y, i_particle, current_t] = epsilon_t*self.kernel(self.delta_values[i_multiple_y, i_particle, current_t]/ epsilon_t)
##########################################################################################################

##########################################################################################################
    # TODO: this function should be an input and not be defined within the framework
    def f_proposal_density_ais(self, theta_mean, i_particle, current_t, sigma):
        return gaussian_density(theta_mean, self.thetas[:self.dim_theta,i_particle, current_t], sigma)
    # TODO: this function should be an input and not be defined within the framework
    def f_proposal_density_sisson(self, theta_mean, i_particle, current_t, sigma):
        raise ValueError("not implemented yet")
        theta_i = self.thetas[:self.dim_theta, i_particle, current_t]
        # need to check the specific form of the density
        return gaussian_density(theta_i, self.thetas[:self.dim_theta, i_particle, current_t], sigma)
##########################################################################################################

##########################################################################################################
    # TODO: this function should be an input and not be defined within the framework
    def f_prior_density(self, i_particle):
        return 1
##########################################################################################################


##########################################################################################################
    def f_weight(self, epsilon_t, current_t, i_particle, theta_mean, sigma):
        """
            function that calculates the weight and switches cases if we initialize the sampler
        """
        if current_t == 0:
            self.f_kernel_value(epsilon_t, current_t, i_particle) # dates kernel_y
            self.kernel_values[:,i_particle, current_t] = self.kernel_values_y[:,i_particle, current_t].mean(axis=0) # save to kernel
        else:
            if self.IS == True:
                self.f_kernel_value(epsilon_t, current_t, i_particle) # dates kernel_y
                self.kernel_values[:,i_particle, current_t] = (self.kernel_values_y[:, i_particle, current_t].mean(axis=0)*self.f_prior_density(i_particle))/self.f_proposal_density_ais(theta_mean, i_particle, current_t, sigma)
            elif self.SMC_Sisson == True:
                self.f_kernel_value(epsilon_t, current_t, i_particle) # dates kernel_y
                self.kernel_values[:,i_particle, current_t] = (self.kernel_values_y[:, i_particle, current_t].mean(axis=0)*self.f_prior_density(i_particle))/self.f_proposal_density_sisson(theta_mean, i_particle, current_t, sigma)

##########################################################################################################

##########################################################################################################
    def f_iterate_weights(self, epsilon_t, current_t, theta_mean, sigma):
        """
            function that returns the weights
        """
        for i_particle in range(self.N_particles):
            self.f_weight(epsilon_t, current_t, i_particle, theta_mean, sigma)

        self.weights[:,:,current_t] = self.kernel_values[:,:,current_t].sum(axis=0)/(np.sum(self.kernel_values[:,:,current_t].flatten()))
        #pdb.set_trace()
##########################################################################################################

##########################################################################################################
    def f_ESS(self, epsilon_t, current_t, theta_mean, sigma):
        """
            function that calculates the ESS
        """
        self.f_iterate_weights(epsilon_t, current_t, theta_mean, sigma)
        self.ESS[:,current_t] = 1/np.sum(self.weights[:,:,current_t]**2)
        return self.ESS[:,current_t]
##########################################################################################################

##########################################################################################################
    def f_dichotomic_search_ESS(self, previous_epsilon, current_t, theta_mean, sigma, N_max_steps=100, tolerance=0.0000001):
        """
            function that does a dichotomic for the root of a function
        """
        n_iter = 0
        eps_inter = previous_epsilon/2.
        eps_left = 0
        eps_right = previous_epsilon*1
        #pdb.set_trace()
        f_inter = self.f_ESS(eps_inter, current_t, theta_mean, sigma)-self.target_ESS
        f_right = self.f_ESS(eps_right, current_t, theta_mean, sigma)-self.target_ESS
        while n_iter<N_max_steps:
            # if same sign on the left side, go right
            eps_outer_right = eps_right
            if np.sign(f_right)==np.sign(f_inter):
                eps_right = eps_inter
                eps_inter = (eps_left+eps_right)/2.
                f_inter = self.f_ESS(eps_inter, current_t, theta_mean, sigma)-self.target_ESS
                f_right = self.f_ESS(eps_right, current_t, theta_mean, sigma)-self.target_ESS
            else:
                eps_left = eps_inter
                eps_inter = (eps_left+eps_right)/2.
                f_inter = self.f_ESS(eps_inter, current_t, theta_mean, sigma)-self.target_ESS
                f_right = self.f_ESS(eps_right, current_t, theta_mean, sigma)-self.target_ESS
            #pdb.set_trace()
            n_iter = n_iter + 1
            if np.abs(f_inter)<tolerance or f_right<0:
                #print "break because of tolerance"
                #print "iterations "+str(n_iter)
                #self.f_ESS(self, eps_inter, current_t, theta_mean, sigma)
                #if f_right<0:
                 #   pdb.set_trace()
                return eps_outer_right
        #print "break because of number of iteration"
        #self.f_ESS(self, eps_inter, current_t, theta_mean, sigma)
        #pdb.set_trace()
        return eps_outer_right
##########################################################################################################


##########################################################################################################
    # first the individual level
    def initialize_sampler(self, theta_sampler):
        """
        function that samples the initial distribution
        """
        # check if we autotune epsilon
        if self.auto_eps == True:
            self.T = self.T_max
            self.epsilon = np.zeros((self.T+1))
        else:
            eps0 = self.epsilon[0]

        print "initialize sampler \n"
        print "starting on "+str(datetime.now())
        print "the run time of the simulation will be approx. in minutes "+str( ((self.T)*self.N_particles*self.multiple_y*0.3)/60)
        # initialize data structure
        self.thetas = np.zeros((self.dim_theta, self.N_particles,self.T))
        self.kernel_values = np.zeros((1, self.N_particles, self.T))
        self.kernel_values_y = np.zeros((self.multiple_y, self.N_particles, self.T))
        self.delta_values = np.zeros((self.multiple_y, self.N_particles, self.T))
        self.weights = np.zeros((1, self.N_particles, self.T)) # one weight per particle
        self.ESS = np.zeros((1,self.T))
        self.ESS_drop = np.zeros((1,self.T))

        i_seed = 0
        for i_particle in range(self.N_particles):
            # create proposal
            theta_prop = theta_sampler(i=i_seed, dim=self.dim_theta, n=1)
            i_seed = 1
            self.thetas[:,i_particle,0] = theta_prop[:,0].transpose()

        # this loop samples several Y for a single theta
            self.f_sample_multiple_y(i_particle, 0, theta_prop)
        if self.auto_eps == True:
            previous_epsilon = 10000000000.
            self.epsilon[0] = self.f_dichotomic_search_ESS(previous_epsilon, 0, theta_mean=None, sigma=None, N_max_steps=100, tolerance=0.001)
            self.f_ESS(self.epsilon[0], 0, theta_mean=None, sigma=None)
        else:
            self.f_ESS(self.epsilon[0], 0, theta_mean=None, sigma=None)
        #pdb.set_trace()
        self.initialized = True
##########################################################################################################


##########################################################################################################
    def f_hilbert_sampling(self, current_t, u):
        """
            hilbert sampling, return the resampled indices
        """
        sorted_u = np.sort(u[:,-1])
        normalised_weigths = self.weights[0,:,current_t]
        normalised_permuted_weights = normalised_weigths[hilbert_sort(self.thetas[:self.dim_theta,:,current_t].transpose())]
        resampled_indices = resampling_inverse_transform(sorted_u, normalised_permuted_weights)
        return resampled_indices
##########################################################################################################

##########################################################################################################
    def f_sample_multiple_y(self, i_particle, current_t, theta_prop):
        """
            function that samples several y and assigns the kernel values
        """
        for i_multiple_y in range(self.multiple_y):
            # sanity check
            if self.exclude_theta(theta_prop)==1:
                y_proposed = self.simulator(self.thetas[:,i_particle, current_t, np.newaxis])
                distance = self.delta(y_proposed, self.y_star)
                pdb.set_trace()
            else :#
                distance = 1000000000

            self.delta_values[i_multiple_y, i_particle, current_t] = distance
##########################################################################################################

##########################################################################################################
    def f_weight_correction(self, current_t, theta_mean, sigma):
        """
            function that is responsible for the weight correction
        """
        #pdb.set_trace()
        if self.auto_eps == True:
            previous_epsilon = self.epsilon[current_t-1]
            self.epsilon[current_t] = self.f_dichotomic_search_ESS(previous_epsilon, current_t, theta_mean, sigma, N_max_steps=100, tolerance=0.001)
            self.f_ESS(self.epsilon[current_t], 0, theta_mean, sigma)
        else:
            for i_particle in range(self.N_particles):
                # renormalize the weights, the N2 comes here !
                # first in case of IS
                if self.IS == True:
                    self.weights[:,i_particle,current_t] = self.kernel_values[:,i_particle,current_t].mean(axis=0)*gaussian_density(theta_mean, self.thetas[:self.dim_theta,i_particle, current_t], sigma)

            # else in smc case
                else:
                    theta_prop = self.thetas[:,i_particle,current_t+1]
                    former_densities = np.array([gaussian_density(theta_prop, self.thetas[:self.dim_theta,i_former_theta, current_t-1], sigma) for i_former_theta in range(self.N_particles)])
                    self.weights[:,i_particle,current_t] = self.kernel_values[:,i_particle, current_t].mean(axis=0)/np.sum(self.weights[:,:, current_t-1]*former_densities*self.kernel_values[:, :, current_t].mean(axis=0))
            # normalize the weights

            nominator = self.weights[:,:,current_t]
            denominator = np.sum(nominator )
            self.weights[:,:,current_t] = nominator/denominator
##########################################################################################################

##########################################################################################################
    def f_ESS_correction_resampling(self, current_t):
        # resampling if ESS to small
        self.ESS[:,current_t] = 1/np.sum(self.weights[:,:,current_t]**2)
        # resampling routine
        if self.ESS[:,current_t] < self.E_treshold:
            self.ESS_drop[:,current_t] = 1
            u_new = self.random_sequence(1, i=0, n=self.N_particles)
            old_weights = self.weights[0,:,current_t] #save the old weights for resampling
            old_thetas =  self.thetas[:,:, current_t] #save the old thetas for resampling
            for i_particle in range(self.N_particles):
                # resampling to get the ancestors
                if self.hilbert == True:
                    pass
                else:
                    pass
                ancestor_new = weighted_choice(old_weights, u_new[i_particle]) # get the ancestor
                theta_resample = old_thetas[:, ancestor_new] # define the old value ( ancestor )
                self.thetas[:,i_particle,current_t] = theta_resample # save the particles
        # resampling done, update the weights
            self.weights[:,:,current_t] = 1./self.N_particles
##########################################################################################################

##########################################################################################################
    def f_kernel_move_particle(self, current_t, theta_mean, sigma, u):

        if self.hilbert == True:
            resampled_indices = self.f_hilbert_sampling(current_t-1, u)

        for i_particle in range(self.N_particles):
            if self.IS == True:
                theta_prop = move_theta(theta_mean, u[i_particle,:self.dim_theta], sigma) # create a completely new particle
            else:
                if self.hilbert==True:
                    ancestor = resampled_indices[i_particle]
                else:
                    ancestor = weighted_choice(self.weights[0,:,current_t-1],u[i_particle,-1]) # get the ancestor
                theta_old = self.thetas[:self.dim_theta,ancestor,current_t-1] # define the old value ( ancestor )
                theta_prop = move_theta(theta_old, u[i_particle,:self.dim_theta], sigma) # move the particle$

            self.thetas[:,i_particle,current_t] = theta_prop # save the particles
            # sample multiple y
            self.f_sample_multiple_y(i_particle, current_t, theta_prop)
##########################################################################################################

##########################################################################################################
    def propogate_particles(self, current_t, move_theta):
        if self.initialized == False:
            raise ValueError('sampler has not been initialized !')
        u = self.random_sequence(self.dim_theta+1, i=0, n=self.N_particles) # random sequence for moves
        # calculate the covariance
        #pdb.set_trace()
        sigma = self.covar_factor*np.cov(self.thetas[:self.dim_theta,:,current_t-1], aweights= np.squeeze(self.weights[:,:,current_t-1]))
    # calculate the mean for IS
        #pdb.set_trace()
        if not is_pos_def(sigma):
            pdb.set_trace()
            raise ValueError('error with the covariance matrix. There could be an error due to the weights')

        theta_mean = np.average(self.thetas[:self.dim_theta,:,current_t-1], axis=1, weights= np.squeeze(self.weights[:,:,current_t-1]))

        self.f_kernel_move_particle(current_t, theta_mean, sigma, u)
        self.f_weight_correction(current_t, theta_mean, sigma)

        # ESS correction and resampling
        #self.f_ESS_correction_resampling(current_t)
##########################################################################################################

##########################################################################################################
    # now the propogation over time
    def loop_over_time(self, move_theta, save=False, name=""):
        for t_index in range(1, self.T):
            print("Now sampling for t %d" % t_index )
            self.propogate_particles(t_index, move_theta)
        if save==True:
            output = {"T" : self.T,
                      "ESS" : self.ESS,
                      "ESS_drop": self.ESS_drop,
                      "covar_factor": self.covar_factor,
                      "dim_theta": self.dim_theta,
                      "E_treshold": self.E_treshold,
                      "IS": self.IS,
                      "N_particles": self.N_particles,
                      "thetas": self.thetas,
                      "weights": self.weights,
                      "y_star": self.y_star,
                      "multiple_y": self.multiple_y,
                      "kernel_values": self.kernel_values,
                      "epsilon": self.epsilon}
            pickle.dump(output, open( name+"_simulation_smc_abc.p", "wb" ))
##########################################################################################################

if __name__ == '__main__':

    if True:
        from functions_tuberculosis_model import *
        ## test tuberculosis model
        N_particles = 100
        epsilon = np.array([1.5,1,0.5,0.2])
    #    smc_abc_mc = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence_mc, uniform_kernel, 2)
    #    smc_abc_mc.initialize_sampler(theta_sampler_mc)
    #    smc_abc_mc.loop_over_time(move_theta)

        smc_abc_rqmc = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence_rqmc, gaussian_kernel, exclude_theta, 2, multiple_y = 3, IS=True, hilbert=True, auto_eps = True, T_max= 100, target_ESS_factor = 0.3)
        smc_abc_rqmc.initialize_sampler(theta_sampler_rqmc)
        #import cProfile
        #cProfile.run('smc_abc_rqmc.loop_over_time(move_theta)')
        smc_abc_rqmc.loop_over_time(move_theta, save=False)
        theta_recovered = np.average( smc_abc_rqmc.thetas[:,:,-1], axis=1, weights= np.squeeze(smc_abc_rqmc.weights[:,:,-1]))
        print theta_recovered

    #a = test.propogate_particles(0,move_theta)
    if False:
        from functions_alpha_stable_model import *
        N_particles = 1000
        #epsilon = np.hstack( (np.linspace(1000,100,10), np.linspace(99,11,89),  np.linspace(10,5,11),  np.linspace(4.95,3.05,39), np.linspace(3,0.01,300)  ))
        epsilon = np.linspace(1000,100,10)
        theta_star = np.array([[1.2],[0.5],[0.1],[0.1]])#theta_sampler_mc_alpha(0)
        #theta_star = theta_sampler_mc_alpha(0)

        y_star = simulator_alpha(theta_star)
        smc_abc_rqmc_alpha = smc_sampler_abc(epsilon, N_particles, delta_alpha, y_star, simulator_alpha, random_sequence_rqmc, gaussian_kernel,covar_factor = 0.25, dim_theta=4, IS=True, exclude_theta=exclude_theta_alpha, hilbert=True, auto_eps = True, T_max= 100, target_ESS_factor = 0.3)
        smc_abc_rqmc_alpha.initialize_sampler(theta_sampler_rqmc_alpha)
        smc_abc_rqmc_alpha.loop_over_time(move_theta)
        theta_recovered = np.average( smc_abc_rqmc_alpha.thetas[:,:,-1], axis=1, weights= np.squeeze(smc_abc_rqmc_alpha.weights[:,:,-1]))
        print theta_recovered
        print smc_abc_rqmc_alpha.ESS
        print smc_abc_rqmc_alpha.epsilon
