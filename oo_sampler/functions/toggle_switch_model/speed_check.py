# speed check toogle switch model

if __name__ == '__main__':
    import numpy as np
    theta = np.array([22, 12, 4, 4.5, 325, 0.25, 0.15], dtype = np.float, ndmin=2)
    import functions_toggle_switch_model
    #pdb.set_trace()
    functions_toggle_switch_model.simulator(theta)
    if True:
        import cProfile
        cProfile.run('functions_toggle_switch_model.repeat_simulator(theta)')
    
    if False:
        import sys
        sys.path.append('/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model/truncated_normal')
        import truncated_normal_a_sample
        def repeat_function(function, N_rep=1000, *args):
            for i in xrange(N_rep):
                print function(*args)
        args = [0,1,0,1]
        print truncated_normal_a_sample.truncated_normal_a_sample(0,1,0,1)
        repeat_function(truncated_normal_a_sample.truncated_normal_a_sample, *args)