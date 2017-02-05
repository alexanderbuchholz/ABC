# -*- coding: utf-8 -*-
assert False # old version ! do not use !
from functions_file import simulator2, theta_sampler_rqmc, delta, theta_sampler_mc
from simulator_p import simulator
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import pickle
from numba import jit

class sample_and_accept():
    """
        class that runs a simulation of the sampler
    """

    def __init__(self, simulator, y_star, theta_sampler, delta, epsilon, dim_theta, type_sim):
        """

        :param simulator:
        :param y_star:
        :param theta_sampler:
        :param delta:
        :param epsilon:
        :return:
        """
        self.simulator = simulator
        self.y_star = y_star
        self.theta_sampler = theta_sampler
        self.epsilon = epsilon
        self.delta = delta
        self.dim_theta = dim_theta
        self.type_sim = type_sim
	print "Attention! This function is depreciated! Are you sure to use this one and not the SMC version??"
    def __call__(self, iteration):
        """

        :param iteration: iteration needed for the Sobol sequence
        :return:
        """
        theta = self.theta_sampler(iteration)
        y_proposed = self.simulator(theta)
        accept = (self.delta(y_proposed, self.y_star) < self.epsilon)
        return [theta, y_proposed, accept]


class loop_simulation(sample_and_accept):
    """

    """
    # function to loop over N
    def run_loop_N(self, N):
        self.sampled_thetas = np.zeros((self.dim_theta, N))
        self.accepted_samples = np.zeros(N)
        bar = progressbar.ProgressBar(maxval=N, \
                                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for n_run in range(N):
            bar.update(n_run + 1)
            inter = self.__call__(n_run)
            self.sampled_thetas[:, n_run] = inter[0].ravel()
            self.accepted_samples[n_run] = inter[2] * 1
        bar.finish()

    # function to loop over batches
    #@jit
    def loop_batches(self,N,batches):
        """

        :param N: number of simulations for one batch
        :param batches: number of batches, number of means that are used for the variance calculation
        :return:
        """
        self.batches_theta = np.zeros((self.dim_theta, N,batches))
        self.accepted_samples_batches = np.zeros((N, batches))
        self.mean_theta = np.zeros((self.dim_theta, batches))
        self.var_theta = np.zeros(self.dim_theta)

        # loop over batches
        for i_batches in range(batches):
            # run the loop N times
            self.run_loop_N(N)
            # save the results
            self.batches_theta[:,:,i_batches] = self.sampled_thetas
            self.accepted_samples_batches[:,i_batches] = self.accepted_samples
            self.mean_theta[:,i_batches] = self.sampled_thetas.dot(self.accepted_samples)/np.sum(self.accepted_samples)
        # calculate the variance
        self.var_theta = np.cov(self.mean_theta)
    # function to loop over list of N

    def loop_list_N(self, N_list, batches, save_data=False):
        """

        :param N_list:
        :param batches:
        :param save_data:
        :return:
        """
        # first create lists that will be filled up
        self.batches_theta_list = []
        self.accepted_samples_batches_list = []
        self.mean_theta_list = []
        self.var_theta_list = []
        # loop over N_list
        for N_entry in N_list:
            # launch loop over one N
            self.loop_batches(N_entry,batches)
            # append results to list
            self.batches_theta_list.append(self.batches_theta)
            self.accepted_samples_batches_list.append(self.accepted_samples_batches)
            self.mean_theta_list.append(self.mean_theta)
            self.var_theta_list.append(self.var_theta)
        output = [self.batches_theta_list, self.accepted_samples_batches_list, self.mean_theta_list, self.var_theta_list]

        if save_data:
            pickle.dump(output, open("tuber_"+str(self.type_sim)+"_simulation_abc_epsilon_"+str(self.epsilon)+".p", "wb") )


y_star = np.array([[1, 282],
                   [2, 20],
                   [3, 13],
                   [4, 4],
                   [5, 2],
                   [8, 1],
                   [10, 1],
                   [15, 1],
                   [23, 1],
                   [30, 1]], dtype=float)

if __name__ == '__main__':
	dim_theta = 2
	epsilon = 1
	# construct class
	tuberculosis_abc = loop_simulation(simulator, y_star, theta_sampler_mc, delta, epsilon, dim_theta, "rqmc")
	import cProfile
	#cProfile.run('tuberculosis_abc.run_loop_N(20)')
	tuberculosis_abc.loop_list_N([5,10],2,True)
	#print tuberculosis_abc.mean_theta
	#print test_loop.accepted_samples
	#plt.plot(test_loop.sampled_thetas[0, test_loop.accepted_samples.astype(bool)],
	#         test_loop.sampled_thetas[1, test_loop.accepted_samples.astype(bool)], 'ro')
	#plt.show()
