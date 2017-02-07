# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:45:28 2016

 Test variational bayes
@author: alex
"""

import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipdb as pdb

from sklearn import mixture
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
import gaussian_densities_etc
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
import f_rand_seq_gen


class vb_sampler():
    """
    a vb sampler,

    """
    def __init__(self, n_components=10, weight_treshold = 0.005, covar_factor=2., sampling="gaussian"):
        self.weight_treshold = weight_treshold
        self.covar_factor = covar_factor
        self.n_components = n_components
        self.vb_estimated = False
        self.sampling = sampling
        if self.sampling == "gaussian":
            self.f_density = gaussian_densities_etc.gaussian_density
            self.f_sample = gaussian_densities_etc.gaussian_move
        elif self.sampling == "t_student":
            self.f_density = gaussian_densities_etc.student_density
            self.f_sample = gaussian_densities_etc.student_move
        else:
            raise ValueError('type of distribution not available')



    def __f_means_weights_tester(self, means=None, covariances=None, weights=None):
        if (self.vb_estimated==False) and (means is not None):
            self.means = means
            self.covariances = covariances
            self.weights = weights
        elif self.vb_estimated == False and (means is None):
            raise ValueError('the sampler has no initial means')

        if np.abs(self.weights.sum()-1.)>0.00001:
            raise ValueError("weights do not sum to one")

    def f_estimate_vb(self, X):
        #pdb.set_trace()
        #dpgmm = mixture.BayesianGaussianMixture(n_components=self.n_components,
        #                                covariance_type='full', weight_concentration_prior_type='dirichlet_distribution',
        #                                weight_concentration_prior = None, n_init = 5, init_params='random', max_iter=800,verbose=1
        #                                ).fit(X.transpose())
        dpgmm = mixture.BayesianGaussianMixture(n_components=self.n_components, max_iter=1000, verbose = 1, verbose_interval = 50, n_init = 5).fit(X.transpose())
        cluster_selector = dpgmm.weights_>self.weight_treshold
        self.means = dpgmm.means_[cluster_selector,:]
        self.covariances = dpgmm.covariances_[cluster_selector,:,:]
        self.weights = dpgmm.weights_[cluster_selector]
        self.weights = self.weights/self.weights.sum()
        self.vb_estimated = True
        #pdb.set_trace()

    def f_vb_mixture_density(self, x, means=None, covariances=None, weights=None):
        """
        function that converts the vb mixture to a density
        the input values must correspond to the results obtained by the dpgmm routine from sklearn
        x : the data point that is evaluated
        means : means of the vb mixture
        covariances : covariances of the vb mixture
        weights : weights of the vb mixture
        weight_tolerance : set threshold for the weights
        """
        # TODO: implement the weights treshold
        self.__f_means_weights_tester(means, covariances, weights)
        #pdb.set_trace()
        n_clusters = self.weights.shape[0]
        dens = 0.
        for i_cluster in range(n_clusters):
            dens = dens + self.weights[i_cluster]*self.f_density(x, self.means[i_cluster,:], self.covar_factor*self.covariances[i_cluster,:,:])
        #pdb.set_trace()
        return dens

    def f_weight_particles(self, particles, means=None, covariances=None, weights=None):
        """
        returns the individual weights of the particles
        """
        #pdb.set_trace()
        N_particles = particles.shape[1]
        weights_particles = np.ones(N_particles)
        for i in xrange(N_particles):
            weights_particles[i] = self.f_vb_mixture_density(particles[:,i])
        return weights_particles

    def f_vb_sampler(self, particles_shape, random_sequence, means=None, covariances=None, weights=None):
        """
        function that simulates according to mixture distribtution
        u : uniform variables that generate the randomness, possibly MC or RQMC
        means : means of mixture components
        covariances = covariance matrices of mxiture components
        weights: weights of mixture components
        covar_factor : set to 1 by default
        weight_treshold : set to 0.05 by default; problem : what happens for the density function ? make one common class ?

        open question : which part of the QMC sequence will be used

        """
        self.__f_means_weights_tester(means, covariances, weights)
        #pdb.set_trace()
        N_particles = particles_shape[1] # shape of u determines the number of particles
        dim_particles = particles_shape[0]
        N_clusters = np.round(self.weights*N_particles)
        particles_next = np.zeros(particles_shape)
        #TODO: problem with rounding !
        # is this the best solution for the rounding problem ??
        #pdb.set_trace()
        while N_clusters.sum() < N_particles:
            N_clusters[np.argmax(self.weights)] += 1
        while N_clusters.sum() > N_particles:
            N_clusters[np.argmin(self.weights)] -= 1
        if N_clusters.sum() != N_particles:
            pdb.set_trace()
            raise ValueError("number ot particles not correct!")
        particle_counter = 0
        cluster_counter = 0
        for j_number_clusters in N_clusters:
            for i_particle in range(int(j_number_clusters)):
                u = random_sequence(size_mv=dim_particles, i=0, n=j_number_clusters)
                #pdb.set_trace()
                particle_prop = self.f_sample(self.means[cluster_counter,:], u[i_particle, :], self.covar_factor*self.covariances[cluster_counter,:,:]) # create a completely new particle
                particles_next[:,particle_counter] = particle_prop # save the particles
                particle_counter += 1
            cluster_counter +=1
        information_components = {'weights': self.weights, 'means':self.means, 'covariances':self.covariances}
        #pdb.set_trace()
        return particles_next, information_components


#
#pdb.set_trace()
#a = f_vb_mixture_density(x, dpgmm.means_, dpgmm.covariances_, dpgmm.weights_, weight_tolerance = 0.05)

if __name__ == '__main__':

    if False:
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                    'darkorange'])


        def plot_results(X, Y_, means, covariances, index, title):
            splot = plt.subplot(2, 1, 1 + index)
            for i, (mean, covar, color) in enumerate(zip(
                    means, covariances, color_iter)):
                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)

            plt.xlim(-9., 5.)
            plt.ylim(-3., 6.)
            plt.xticks(())
            plt.yticks(())
            plt.title(title)


        # Number of samples per component
        n_samples = 500

        # Generate random sample, two components
        np.random.seed(0)
        C = np.array([[0., -0.1], [1.7, .4]])
        '''X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                .7 * np.random.randn(n_samples, 2) + np.array([-2, 3])]
        X = np.r_[np.dot(np.random.randn(n_samples, 2), 3)+ np.array([-6, 3]),
                .1 * np.random.randn(n_samples, 2)]'''

        X = np.r_[np.random.randn(n_samples, 2),
                .01 * np.random.randn(n_samples, 2)]

        # Fit a Gaussian mixture with EM using five components
        gmm = mixture.GaussianMixture(n_components=10, covariance_type='full').fit(X)
        plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                    'Gaussian Mixture')

        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=4,
                                                covariance_type='full').fit(X)
        plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                    'Bayesian Gaussian Mixture with a Dirichlet process prior')
        plt.show()
        pdb.set_trace()


    n1 = 100
    n2 = 100
    n3 = 100
    n = n1 + n2+n3
    dim = 4
    i = 0
    simulations = 10000
    #u = np.random.random(size=(n,dim+1))
    #u = random_sequence_rqmc(size_mv = dim+1, i= i, n = n)
    test1 = np.random.normal(size=(n1, dim))*1
    test2 = np.random.normal(size=(n2, dim))*1# + np.array([2,2])
    test3 = np.random.normal(size=(n3, dim))*1# + np.array([-1,2])
    #proposal_values = np.vstack((test1, test2, test3))
    #from scipy.stats import t as student
    #proposal_values = np.reshape(student.rvs(5, size=simulations*dim), (simulations, dim))*2
    proposal_values = np.random.normal(size=(simulations, dim))*2
    pdb.set_trace()
    def f_weights_mixture(x, target_only=False, dim=2):
        """
        importance sampling for the gaussian mixture model
        """
        mu = np.zeros(dim)
        sigma = np.eye(dim)
        weights = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            #pdb.set_trace()      
            weight_mixture = 0.5*gaussian_densities_etc.gaussian_density(x[i, :], mu, sigma) +0.5*gaussian_densities_etc.gaussian_density(x[i, :], mu, 0.01*sigma)
            if target_only == False:
                weight_student = gaussian_densities_etc.gaussian_density(x[i, :], mu, sigma*2)
                weights[i] = weight_mixture/weight_student
            else: 
                weights[i] = weight_mixture
        if target_only == False:
            weights_normalized = weights/weights.sum()
            return weights_normalized
        else: 
            return weights
        
        
    
    weights = f_weights_mixture(proposal_values, dim=dim)
    import resampling
    indexes = resampling.stratified_resample(weights)
    test = proposal_values[indexes, :]
    vb_sampler = vb_sampler(n_components=10, covar_factor=1, sampling="gaussian")
    vb_sampler.f_estimate_vb(test.transpose())
    def f_ESS_target_proposal(particles, dim, vb_sampler):
        weights_target_after_resampling = f_weights_mixture(particles, target_only=True, dim=dim)
        weights_proposal_after_resampling = vb_sampler.f_weight_particles(particles.transpose())
        weights_after_resampling = weights_target_after_resampling/weights_proposal_after_resampling
        weights_new = weights_after_resampling/weights_after_resampling.sum()
        ESS = 1/sum(weights_new**2)
        print ESS
        return ESS

    f_ESS_target_proposal(test, dim, vb_sampler)
    
    #vb_sampler.f_vb_mixture_density(x)
    random_sequence = f_rand_seq_gen.random_sequence_rqmc
    particles_shape = (dim, simulations)
    new_samples, info_components = vb_sampler.f_vb_sampler(particles_shape, random_sequence)
    
    new_samples = new_samples.transpose()
    f_ESS_target_proposal(new_samples, dim, vb_sampler)
    pdb.set_trace()
    #pdb.set_trace()
    if True:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        #lim = (-0.5, 0.5)
        x1_test = pd.Series(test[:, 0], name="$X_1$")
        x2_test = pd.Series(test[:, 1], name="$X_2$")
        sns.jointplot(x1_test, x2_test, kind="kde")#, xlim = lim, ylim = lim)

        x1_new = pd.Series(new_samples[:, 0], name = "$X_1$")
        x2_new = pd.Series(new_samples[:, 1], name = "$X_2$")
        sns.jointplot(x1_new, x2_new, kind = "kde")#, xlim = lim, ylim = lim)
        #plt.scatter(test[:,0], test[:,1], color='red')
        #plt.scatter(test2[:,0], test2[:,1], color='green')
        #plt.scatter(test3[:,0], test3[:,1], color='blue')
        #plt.scatter(new_samples[:,0], new_samples[:,1], color='black')
        plt.show()
    pdb.set_trace()