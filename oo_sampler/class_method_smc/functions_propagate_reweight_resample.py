# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:13:32 2016
functions that define how to propagate, reweight and resample
@author: alex
"""
import numpy as np
import ipdb as pdb
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/hilbert")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/propagater_smc/VB")
import f_rand_seq_gen
import gaussian_densities_etc
import function_vb_class_sampler
from functools import partial
import functions_transform_hilbert
import resampling
#from joblib import Parallel, delayed
import pathos.multiprocessing as multiprocessing
NUM_CORES = multiprocessing.cpu_count()
from scipy.stats import multivariate_normal

class simulator_sampler():
    """
    class that is responsible for the simulation
    """
    def __init__(self, simulator, y_star, delta, exclude_theta, M_simulator, parallelize=False):
        self.simulator = simulator
        self.y_star = y_star
        self.delta = delta
        self.exclude_theta = exclude_theta
        self.parallelize = parallelize
        self.M_simulator = M_simulator

    def f_parallel_loop(self, n_particle, particles):
        """
        parallel simulation
        we directly return the value of the distance, this is a scalar !
        """
        #pdb.set_trace()
        particle_n = particles[:, n_particle, np.newaxis]
        aux_particles_indvidual = np.zeros((self.M_simulator, 1))
        if self.exclude_theta(particle_n) == 1:
            # if ok sample
            #pdb.set_trace()
            for m_particle in xrange(self.M_simulator):
                y_proposed = self.simulator(particle_n)
                distance = self.delta(y_proposed, self.y_star)
                aux_particles_indvidual[m_particle, :] = distance
                # if not set large distance
        else:
            aux_particles_indvidual = np.ones((self.M_simulator, 1))*10000000000.
        return aux_particles_indvidual

    def f_parallel_loop_negative_binomial(self, n_particle, particles, epsilon_target, n_successfull_tries=2):
        """
        parallel simulation
        we directly return the value of the distance, this is a scalar !
        return the two smallest distances that are below the threshold and 
        the number of tries necessary
        """
        #pdb.set_trace()
        particle_n = particles[:, n_particle, np.newaxis]
        aux_particles_indvidual = np.zeros((n_successfull_tries, 1))
        aux_particles_tries = np.zeros(1)
        if self.exclude_theta(particle_n) == 1:
            # if ok sample
            #pdb.set_trace()
            achieved_successful_tries = 0
            while achieved_successful_tries < n_successfull_tries:
                y_proposed = self.simulator(particle_n)
                distance = self.delta(y_proposed, self.y_star)
                if distance < epsilon_target:
                    aux_particles_indvidual[achieved_successful_tries, :] = distance
                    achieved_successful_tries += 1
                aux_particles_tries += 1.
            aux_particles_tries_until_success = aux_particles_tries - n_successfull_tries
        else:
            aux_particles_indvidual = np.ones((n_successfull_tries, 1))*10000000000.
            aux_particles_tries_until_success = np.inf
        #pdb.set_trace()
        return aux_particles_indvidual, aux_particles_tries_until_success


    def f_auxialiary_sampler(self, particles, *args, **kwargs):
        """
            function that samples according to the size of the particles
        """
        #pdb.set_trace()
        N_particles = particles.shape[1]
        aux_particles = np.zeros((self.M_simulator, N_particles))
        # loop over the particles (theta)
        partial_f_parallel_loop = partial(self.f_parallel_loop, particles=particles)
        if self.parallelize == True:
            pool = multiprocessing.Pool(processes=(NUM_CORES))
            F = pool.map(partial_f_parallel_loop, range(N_particles))
            pool.close()
            pool.join()
            for n_particle in xrange(N_particles):
                aux_particles[:, n_particle] = F[n_particle].squeeze()
        else:
            for n_particle in xrange(N_particles):
                # check if theta is admissible
                aux_particles_ind = partial_f_parallel_loop(n_particle)
                aux_particles[:, n_particle] = aux_particles_ind.squeeze()

        gaussian_densities_etc.break_if_nan(aux_particles)
        return aux_particles

    def f_auxialiary_sampler_negative_binomial(self, particles, epsilon_target, n_successfull_tries=2):
        """
        negative binomial sampling
        """
        
        """if True:
            pdb.set_trace()
            from matplotlib import pyplot as plt
            plt.hist(particles.flatten())
            plt.show()"""
        N_particles = particles.shape[1]
        aux_particles = np.zeros((n_successfull_tries, N_particles))
        aux_particles_tries = np.zeros((1, N_particles))
        # loop over the particles (theta)
        partial_f_parallel_loop = partial(self.f_parallel_loop_negative_binomial, particles=particles, epsilon_target=epsilon_target, n_successfull_tries=n_successfull_tries)
        if self.parallelize == True:
            pool = multiprocessing.Pool(processes=(NUM_CORES))
            F = pool.map(partial_f_parallel_loop, range(N_particles))
            pool.close()
            pool.join()
            for n_particle in xrange(N_particles):
                #pdb.set_trace()
                aux_particles[:, n_particle] = F[n_particle][0].squeeze()
                aux_particles_tries[:,n_particle] = F[n_particle][1]
        else:
            #pdb.set_trace()
            for n_particle in xrange(N_particles):
                # check if theta is admissible
                aux_particles_indvidual, aux_particles_tries_until_success = partial_f_parallel_loop(n_particle)
                aux_particles[:, n_particle] = aux_particles_indvidual.squeeze()
                aux_particles_tries[:,n_particle] = aux_particles_tries_until_success

        gaussian_densities_etc.break_if_nan(aux_particles)
        return aux_particles, aux_particles_tries


def f_hilbert_sampling(particles, weights, u):
    """
        hilbert sampling, return the resampled indices
    """
    sorted_u = np.sort(u[:,-1])
    normalised_permuted_weights = weights[:,functions_transform_hilbert.hilbert_sort(particles.transpose())].squeeze()
    resampled_indices = functions_transform_hilbert.resampling_inverse_transform(sorted_u, normalised_permuted_weights)
    return resampled_indices


class propagater_particles():
    """
    class that is responsible for the propagation of the particles
    """
    def __init__(self, dim_particles, N_particles, move_particle, propagation_mechanism="AIS", sampler_type="MC", covar_factor = 1., ancestor_sampling=False, mixture_components=1):
        self.propagation_mechanism = propagation_mechanism
        self.ancestor_sampling = ancestor_sampling
        self.sampler_type = sampler_type
        self.dim_particles = dim_particles
        self.N_particles = N_particles
        self.mixture_components = mixture_components
        self.move_particle = move_particle # a function
        self.covar_factor = covar_factor

    def f_propagate_ais(self, random_sequence, particles, weights, flag_failed_ESS, information_components, *args, **kwargs):
        """
        propagation of type ais
        """
        if self.mixture_components == 1:
            #pdb.set_trace()
            particles_var = np.atleast_2d(self.covar_factor*np.cov(particles, aweights=np.squeeze(weights)))
            particles_mean = np.average(particles, weights=np.squeeze(weights), axis=1)
            u = random_sequence(self.dim_particles, i=0, n=self.N_particles)
            particles_next = np.zeros(particles.shape)
            particles_preweights = np.ones(weights.shape)
            #pdb.set_trace()
            for i_particle in range(self.N_particles):
                # resampling first, to pick the ancestors
                particles_next[:,i_particle] = self.move_particle(particles_mean, u[i_particle,:], particles_var) # move the particle$
                # calculate the weights
                #pdb.set_trace()
                density_particle = np.array([gaussian_densities_etc.gaussian_density(particles_next[:,i_particle], particles_mean, particles_var)])
                density_prior = 1 # TODO: add real prior
                weight_inter = density_prior/density_particle
                if np.isnan(weight_inter).any():
                    raise ValueError('some particles is Nan!')
                    weight_inter = 0.
                particles_preweights[:,i_particle] = weight_inter
            return particles_next, particles_preweights, information_components.append([particles_mean, particles_var])
        else: 
            particles_resampled = np.zeros(particles.shape)
            # implement residual resampling
            ancestors = resampling.residual_resample(np.squeeze(weights))
            particles_resampled = particles[:, ancestors] # define the old value ( ancestor )
            vb_sampler = function_vb_class_sampler.vb_sampler(n_components = self.mixture_components, covar_factor=self.covar_factor)

            if flag_failed_ESS == True and len(information_components) != 0:
                # use the old values of the previous iteration
                #pdb.set_trace()
                vb_sampler.weights = information_components[-1]["weights"]
                vb_sampler.means = information_components[-1]["means"]
                vb_sampler.covariances = information_components[-1]["covariances"]
                vb_sampler.vb_estimated = True

            else:
                vb_sampler.f_estimate_vb(particles_resampled)
            
            particles_next, information_components = vb_sampler.f_vb_sampler(particles.shape, random_sequence)
            #pdb.set_trace()
            particles_preweights_proposal = vb_sampler.f_weight_particles(particles_next)
            prior = 1.
            particles_preweights = prior/particles_preweights_proposal
            # TODO: a real prior is missing !
            return particles_next, particles_preweights, information_components

    def f_propagate_nonparametric(self, particles, weights, u, *args, **kwargs):
        """
        propagation as done by sisson
        """
        particles_var = self.covar_factor*np.cov(particles, aweights=np.squeeze(weights))
        particles_var = np.atleast_2d(particles_var)
        #pdb.set_trace()
        if self.ancestor_sampling == "Hilbert":
            #pdb.set_trace()
            resampled_indices = f_hilbert_sampling(particles, weights, u)
        else:
            resampled_indices = resampling.residual_resample(np.squeeze(weights))

        particles_old = np.zeros(particles.shape)
        particles_next = np.zeros(particles.shape)
        particles_preweights = np.ones(weights.shape)

        for i_particle in range(self.N_particles):
            # resampling first, to pick the ancestors
            ancestor = int(resampled_indices[i_particle])
            particles_old[:, i_particle] = particles[:, ancestor] # define the old value (ancestor)
            particle_prop = self.move_particle(particles_old[:,i_particle], u[i_particle,:-1], particles_var) # move the particle$
            particles_next[:, i_particle] = particle_prop # save the particles
            # calculate the weights

        for i_particle in range(self.N_particles):
            density_particle = (weights*multivariate_normal.pdf(particles_old.transpose(), mean=particles_next[:,i_particle], cov=particles_var)).transpose()
            density_prior = 1 # TODO: add real prior
            weight_inter = density_prior/density_particle.sum()
            if np.isnan(weight_inter).any():
                raise ValueError('some particles are Nan!')
                weight_inter = 0.
            particles_preweights[:, i_particle] = weight_inter


        
        return particles_next, particles_preweights, []

    def f_propagate_del_moral(self, particles, weights, u, *args, **kwargs):
        """
        propagation as done by del moral
        """
        particles_next = np.zeros(particles.shape)
        #pdb.set_trace()
        particles_var = self.covar_factor*np.cov(particles, aweights=np.squeeze(weights))
        particles_preweights = np.ones(weights.shape)

        for i_particle in range(self.N_particles):
            particle_prop = self.move_particle(particles[:, i_particle], u[i_particle, :-1], particles_var) # move the particle$
            particles_next[:, i_particle] = particle_prop # save the particles
            # calculate the weights
        return particles_next, particles_preweights, []
    #######################################################################################


    def f_propagate(self, particles, weights, flag_failed_ESS, information_components, *args, **kwargs):
        """
        function that is responsible for the propagation of the particles, based on the previous particles
        """
        # chose if we need an additional random number for the ancestor selection
        gaussian_densities_etc.break_if_nan(particles)
        gaussian_densities_etc.break_if_nan(weights)
        particles_next = np.zeros(particles.shape)
        if self.sampler_type=="MC":
            random_sequence = f_rand_seq_gen.random_sequence_mc
        elif self.sampler_type=="RQMC":
            random_sequence = f_rand_seq_gen.random_sequence_rqmc
        elif self.sampler_type=="QMC":
            random_sequence = f_rand_seq_gen.random_sequence_qmc
        else: 
            raise ValueError("sampler type not specified or unavailable")
        # generate the random sequence
        # choose which sample mechanism to choose
        #########################################################################################
        if self.propagation_mechanism == 'AIS':
            """
            AIS
            """
            # TODO: add a real prior
            # pass the class for the reweighting scheme
            particles_next, particles_preweights, information_components = self.f_propagate_ais(random_sequence, particles, weights, flag_failed_ESS, information_components, *args, **kwargs)
            """if True:
                pdb.set_trace()
                import matplotlib.pyplot as plt
                import seaborn as sns
                #for dim in range(7):   
                dim = 0 
                sns.distplot(particles[dim, :], label='current')
                sns.kdeplot(particles_next[dim, :], label='next')
                plt.show()
                pdb.set_trace()"""
        #########################################################################################

        #########################################################################################
        elif self.propagation_mechanism == 'nonparametric':
            """
            nonparametric a la sisson
            """
            u = random_sequence(self.dim_particles+1, i=0, n=self.N_particles)
            particles_next, particles_preweights, information_components = self.f_propagate_nonparametric(particles, weights, u, *args, **kwargs)
        #########################################################################################
        elif self.propagation_mechanism == 'Del_Moral':
            """
            del moral adaptive smc with MH move
            """
            u = random_sequence(self.dim_particles+1, i=0, n=self.N_particles)
            particles_next, particles_preweights, information_components = self.f_propagate_del_moral(particles, weights, u, *args, **kwargs)
        #########################################################################################

        return particles_next, particles_preweights, information_components
#########################################################################################
#########################################################################################

def calculate_weights_del_moral(epsilon, particles_preweights, aux_particles, weights_before, aux_particles_tries_current_t=[], kernel=gaussian_densities_etc.uniform_kernel, previous_epsilon=None):
    """
    the function that calculates the weights in the approach of del moral
    """
    density_aux_nominator = gaussian_densities_etc.f_kernel_value(epsilon, aux_particles, kernel).mean(axis=0)
    density_aux_denominator = gaussian_densities_etc.f_kernel_value(previous_epsilon, aux_particles, kernel).mean(axis=0)
    weights_inter = density_aux_nominator/density_aux_denominator
    weights_next = weights_inter*weights_before
    eliminator = np.isnan(weights_next)
    weights_next[eliminator] = 0.
    gaussian_densities_etc.break_if_nan(weights_next)
    #pdb.set_trace()
    return weights_next

def accept_reject_del_moral(epsilon, aux_particles_new, aux_particles_old, kernel=gaussian_densities_etc.uniform_kernel):
    N_particles = aux_particles_new.shape[1] # the shape of the weights is (1,N) !!!!!
    density_aux_nominator = gaussian_densities_etc.f_kernel_value(epsilon, aux_particles_new, kernel).mean(axis=0)
    density_aux_denominator = gaussian_densities_etc.f_kernel_value(epsilon, aux_particles_old, kernel).mean(axis=0)
    weights_next = density_aux_nominator/density_aux_denominator
    eliminator = np.isnan(weights_next)
    weights_next[eliminator] = 0.

    gaussian_densities_etc.break_if_nan(weights_next)
    #pdb.set_trace()
    probabilities = np.random.uniform(size=N_particles)
    return (weights_next > probabilities)

def calculate_weights(epsilon, particles_preweights, aux_particles, weights_before=None, aux_particles_tries_current_t=[], kernel=gaussian_densities_etc.gaussian_kernel, previous_epsilon=None):
    """
    general weight calculation
    """
    # vectorize code
    """pdb.set_trace()
    if False: 
        from matplotlib import pyplot as plt
        aux_particles_inter = aux_particles.flatten()
        plt.hist(aux_particles_inter[aux_particles_inter<100000])
        plt.show()
    """
    if len(aux_particles_tries_current_t)>0: # list is not empty, use it for the negative binomial model
        density_aux = 1./(aux_particles_tries_current_t+1.)
    else:
        density_aux = gaussian_densities_etc.f_kernel_value(epsilon, aux_particles, kernel).mean(axis=0)
    weights_next = particles_preweights*density_aux
    eliminator = np.isnan(weights_next)
    weights_next[eliminator] = 0.
    gaussian_densities_etc.break_if_nan(weights_next)
    #pdb.set_trace()
    return weights_next

def calculate_weights_ESS(epsilon, f_calculate_weights, *args, **kwargs):
    """
    wrapper function for the ESS calculation
    """
    weights_next = f_calculate_weights(epsilon, *args, **kwargs)
    weights_normalized = weights_next/np.sum(weights_next)
    ESS = 1/(weights_normalized**2).sum()
    return ESS

def reweight_particles(epsilon, f_calculate_weights, particles_next, particles_preweights, covar_factor, aux_particles, weights_before, aux_particles_tries_current_t=[], kernel=gaussian_densities_etc.gaussian_kernel, previous_epsilon=None):
    """
    implmenents the reweighting scheme
    """

    weights_next = f_calculate_weights(epsilon, particles_preweights, aux_particles, weights_before, aux_particles_tries_current_t, kernel, previous_epsilon)


    weights_normalized = weights_next/np.sum(weights_next)
    gaussian_densities_etc.break_if_nan(weights_normalized)
    gaussian_densities_etc.break_if_negative(weights_normalized)
    ESS = 1/(weights_normalized**2).sum()
    #pdb.set_trace()

    particles_mean_next = np.average(particles_next, axis=1, weights=np.squeeze(weights_normalized))
    particles_var_next = covar_factor*np.cov(particles_next, aweights=np.squeeze(weights_normalized))
    #raise ValueError('this function should not be used anymore !')
    return weights_normalized, particles_mean_next, particles_var_next, ESS



    ###################################################################################################################################
    ###################################################################################################################################


def f_dichotomic_search_ESS(previous_epsilon, partial_f_ESS, target_ESS, N_max_steps=100, tolerance=0.1):
    """
        function that does a dichotomic for the root of a function
    """
    n_iter = 0
    eps_inter = previous_epsilon/2.
    eps_left = 0
    eps_right = previous_epsilon*1
    #pdb.set_trace()
    f_inter = partial_f_ESS(eps_inter)-target_ESS
    f_right = partial_f_ESS(eps_right)-target_ESS
    while n_iter<N_max_steps:
        # if same sign on the left side, go right
        eps_outer_right = eps_right
        #pdb.set_trace()
        if np.sign(f_right)==np.sign(f_inter):
            eps_right = eps_inter
            eps_inter = (eps_left+eps_right)/2.
            f_inter = partial_f_ESS(eps_inter)-target_ESS
            f_right = partial_f_ESS(eps_right)-target_ESS
        else:
            eps_left = eps_inter
            eps_inter = (eps_left+eps_right)/2.
            f_inter =  partial_f_ESS(eps_inter)-target_ESS
            f_right = partial_f_ESS(eps_right)-target_ESS
        #pdb.set_trace()
        n_iter = n_iter + 1
        if np.abs(f_inter)<tolerance or f_right<0:
             #   pdb.set_trace()
            return eps_outer_right
    #pdb.set_trace()
    return eps_outer_right

###########################################################################################################################
class reweighter_particles():
    """
    class that is responsible for the propagation of the particles
    """
    def __init__(self, 
                    dim_particles, 
                    N_particles, 
                    propagation_mechanism="AIS", 
                    covar_factor = 1., 
                    ancestor_sampling=False, 
                    mixture_components=1, 
                    target_ESS_ratio=0.3, 
                    autochoose_eps = False, 
                    kernel=gaussian_densities_etc.gaussian_kernel, 
                    alpha=0.90,
                    epsilon_target = None):
        self.propagation_mechanism = propagation_mechanism
        self.dim_particles = dim_particles
        self.N_particles = N_particles
        self.covar_factor = covar_factor
        self.target_ESS_reweighter = target_ESS_ratio*N_particles
        self.autochoose_eps = autochoose_eps
        self.kernel = kernel
        self.alpha = alpha
        self.epsilon_target = epsilon_target
        #self.quantile_target = quantile_target


    def f_reweight(self, particles_next, particles_preweights, current_t, epsilon, aux_particles, weights_before, aux_particles_tries_current_t=[], previous_ESS=None, quantile_target = 0.8, **kwargs):
        """
        function that is responsible for the reweigthing of the particles, based on the previous particles
        """
        # chose if we need an additional random number for the ancestor selection
        gaussian_densities_etc.break_if_nan(particles_next)
        gaussian_densities_etc.break_if_nan(particles_preweights)
        gaussian_densities_etc.break_if_nan(aux_particles)

        self.N_particles = particles_next.shape[1]
        # generate the random sequence
        # choose which sample mechanism to choose
        if current_t == 0:
            previous_epsilon = 100000
        else:
            previous_epsilon = epsilon[current_t-1]

        if self.propagation_mechanism == "Del_Moral":
            previous_epsilon_del_moral = previous_epsilon
            target_ESS = self.alpha*previous_ESS
            f_calculate_weights = calculate_weights_del_moral

        else:
            previous_epsilon_del_moral = None
            target_ESS = self.target_ESS_reweighter
            f_calculate_weights = calculate_weights

        if target_ESS is None: raise ValueError('target ESS is none !!')
        #pdb.set_trace()
        partial_f_ESS = partial(calculate_weights_ESS,
                                f_calculate_weights = f_calculate_weights,
                                particles_preweights=particles_preweights,
                                aux_particles = aux_particles,
                                weights_before = weights_before,
                                aux_particles_tries_current_t = aux_particles_tries_current_t,
                                kernel = self.kernel, 
                                previous_epsilon = previous_epsilon_del_moral)
        
    
        if self.autochoose_eps == 'quantile_based':
            #pdb.set_trace()
            M = aux_particles.shape[0]
            #aux_particles_means =  aux_particles.mean(axis=0)
            quantile_index = np.round(M*self.N_particles*quantile_target)
            
            #epsilon_proposed = np.sort(aux_particles_means)[quantile_index]
            number_inadmissable_particles = sum(aux_particles.flatten()>1000)
            epsilon_proposed = np.sort(aux_particles.flatten())[quantile_index-number_inadmissable_particles]
            epsilon_current = np.min(np.array([epsilon_proposed, previous_epsilon]))
            #pdb.set_trace()
            
        elif self.autochoose_eps == 'ess_based':
            # routine for autochosing epsilon based on bisect search
            epsilon_current = f_dichotomic_search_ESS(previous_epsilon, partial_f_ESS, target_ESS)
            if epsilon_current < self.epsilon_target:
                epsilon_current = self.epsilon_target
            
            if False:
                import matplotlib.pyplot as plt
                epsilon_values = np.linspace(0.05, 100, num=100)
                ESS_values = np.zeros(epsilon_values.shape)
                for i in range(len(epsilon_values)):
                    ESS_values[i] = partial_f_ESS(epsilon_values[i])
                plt.plot(epsilon_values, ESS_values)
                plt.show()
            #pdb.set_trace()
        else:
            epsilon_current = epsilon[current_t]

        ESS_before_reweighting = partial_f_ESS(previous_epsilon)
        if ESS_before_reweighting < self.target_ESS_reweighter:
            print 'ESS breakdown!'
        #pdb.set_trace()
        variances_normalisation_constant = gaussian_densities_etc.f_kernel_value(epsilon_current, aux_particles, self.kernel).var(axis=0)
        means_normalisation_constant = gaussian_densities_etc.f_kernel_value(epsilon_current, aux_particles, self.kernel).mean(axis=0)
        weights_next, particles_mean, particles_var, ESS = reweight_particles(epsilon_current,
                                                                                      f_calculate_weights = calculate_weights,
                                                                                      particles_next = particles_next,
                                                                                      particles_preweights = particles_preweights,
                                                                                      covar_factor = self.covar_factor,
                                                                                      aux_particles = aux_particles,
                                                                                      weights_before = weights_before, 
                                                                                      aux_particles_tries_current_t = aux_particles_tries_current_t, 
                                                                                      kernel = self.kernel,
                                                                                      previous_epsilon = None)
        print "ESS = %s, ESS before reweighting = %s, current epsilon = %s" %(ESS, ESS_before_reweighting, epsilon_current)
        return weights_next, particles_mean, particles_var, ESS, epsilon_current, ESS_before_reweighting, variances_normalisation_constant, means_normalisation_constant
###########################################################################################################################
class resampler_particles():
    """
    """
    def __init__(self, N_particles):
        self.N_particles = N_particles
        #self.propagation_mechanism = propagation_mechanism
    def f_resampling(self, weights_current, particles_current, ESS, ESS_treshold, **kwargs):
        # resampling routine
        gaussian_densities_etc.break_if_nan(weights_current)
        gaussian_densities_etc.break_if_nan(particles_current)
        #pdb.set_trace()
        particles_resampled = np.zeros(particles_current.shape)
        if ESS < ESS_treshold:
            print "Resample since ESS %d is smaller than treshold %d" %(ESS, ESS_treshold)
            ancestors = resampling.residual_resample(np.squeeze(weights_current))
            particles_resampled = particles_current[:, ancestors]
            gaussian_densities_etc.break_if_nan(particles_resampled)
            """if True: 
                pdb.set_trace()
                from matplotlib import pyplot as plt
                plt.hist(particles_resampled.flatten())
                plt.show()
                pdb.set_trace()"""
            return particles_resampled, np.ones(weights_current.shape)/self.N_particles
        else:
            return particles_current, weights_current
###########################################################################################################################

if __name__=='__main__':
    if True:
        import sys
        sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model")
        import functions_mixture_model
        theta = functions_mixture_model.theta_sampler_rqmc(0, 2, 10)
        print theta
        y_star = np.zeros(2)
        simulator_mm = simulator_sampler(functions_mixture_model.simulator,
                                         y_star,
                                         functions_mixture_model.delta,
                                         functions_mixture_model.exclude_theta)
        kwargs = {'M_simulator':5}
        dist = simulator_mm.f_auxialiary_sampler(theta, **kwargs)
        print dist.shape
        print dist

    if False:
        sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
        sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
        import functions_tuberculosis_model
        theta = functions_tuberculosis_model.theta_sampler_rqmc(0, 2, 10)
        print theta
        simulator_tuber = simulator_sampler(functions_tuberculosis_model.simulator,
                                         functions_tuberculosis_model.y_star,
                                         functions_tuberculosis_model.delta,
                                         functions_tuberculosis_model.exclude_theta)
        dist = simulator_tuber.f_auxialiary_sampler(theta)
        print dist.shape
        print dist
