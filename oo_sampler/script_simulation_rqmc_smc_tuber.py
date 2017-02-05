from functions_smc import *
import pickle

N_particles = 2500
#epsilon = np.array([2.,1.])
if True:
	epsilon = np.array([2.,1.5,1,0.5,0.2,0.1,0.08,0.065,0.05,0.04,0.03,0.02,0.01])
	smc_abc_rqmc = smc_sampler_abc(epsilon, N_particles, delta, y_star, simulator, random_sequence_rqmc, uniform_kernel, 2)
	smc_abc_rqmc.initialize_sampler(theta_sampler_rqmc)
	smc_abc_rqmc.loop_over_time(move_theta, save=True, name="rqmc_run1")

	pickle.dump( smc_abc_rqmc, open( "run1_total_rqmc_smc_abc.p", "wb" ) )
	#smc_abc_rqmc = pickle.load( open( "test_rqmc_smc_abc.p", "rb" ) )
