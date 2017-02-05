from functions_smc import *
import pickle

N_particles = 2500
#epsilon = np.array([2.,1.])
if True:
	epsilon = np.array([2.,1.5,1,0.5,0.2,0.1,0.08,0.065,0.05,0.04,0.03,0.02,0.01])

	smc_abc_mc = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence_mc, uniform_kernel, 2)
	smc_abc_mc.initialize_sampler(theta_sampler_mc)
	smc_abc_mc.loop_over_time(move_theta, save = True, name="mc_run1")

	pickle.dump( smc_abc_mc, open( "run1_total_mc_smc_abc.p", "wb" ) )
#
