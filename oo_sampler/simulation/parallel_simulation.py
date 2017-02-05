# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:56:45 2017
    parallel simulation of ABC SMC
@author: alex
"""

import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/class_method_smc")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/simulation")
import functions_propagate_reweight_resample
import class_smc

NUM_PROCESSES = 2
import ipdb as pdb
def set_up_parallel_abc_sampler(filename, **kwargs):
    '''
        function responsible for setting up the sampler and launching the simulation
    '''
    globals().update(**kwargs)
    #   pdb.set_trace()
    for N_particles in N_particles_list:
        #pass
        test_sampler = class_smc.smc_sampler(N_particles, 
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
        test_sampler.setInitiationFunction(inititation_particles)
        test_sampler.propagation_mechanism = propagation_mechanism
        test_sampler.sampler_type = sampler_type
        test_sampler.covar_factor = covar_factor
        import functions_propagate_reweight_resample

        simulator_mm = functions_propagate_reweight_resample.simulator_sampler(simulator,
                                     y_star,
                                     delta,
                                     exclude_theta,
                                     M_simulator = dim_auxiliary_var)
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
        test_sampler.setEpsilonSchedule(epsilon)
        test_sampler.setReweightFunction(reweighter.f_reweight)
    #test_sampler.reweight_particles(0)
        resampler = functions_propagate_reweight_resample.resampler_particles(N_particles)
        test_sampler.setResampleFunction(resampler.f_resampling)
        test_sampler.iterate_smc(resample=resample, save=save, filename=filename, modified_sampling=propagation_mechanism)

if True:
    from multiprocessing import Process, Pipe
    from itertools import izip
    from functools import partial

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

