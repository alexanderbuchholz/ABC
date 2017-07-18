def simulation_joint_distribution(simulator, delta, theta_sampler, dim, N_simulations, y_star, simulator_vectorized=False):
    """
    a function that simulates the static joint distribution,
    contains a vectorized version that makes the simulation much faster

    returns the prior values and the associated distances
    """
    prior_values = theta_sampler(0, dim, N_simulations)
    distances = np.zeros(N_simulations)
    #pdb.set_trace()
    if simulator_vectorized:
        y_pseudo = simulator(prior_values)
        distances = delta(y_star, y_pseudo)
        #pdb.set_trace()
    else:
        for iteration in range(N_simulations):
            y_pseudo = simulator(prior_values[:, iteration])
            distances[iteration] = delta(y_pseudo, y_star)
    return(prior_values, distances)


def repeat_joint_simulation(M_repetitions, simulator, delta, theta_sampler, dim, N_simulations, y_star, simulator_vectorized=False):
    """
    iterates the simulation such that we can asses the posterior variance 
    """
    reference_table_theta = np.zeros((dim, N_simulations, M_repetitions))
    reference_table_distances = np.zeros((1, N_simulations, M_repetitions))
    for m_iteration in range(M_repetitions):
        prior_values, distances = simulation_joint_distribution(
                            simulator, 
                            delta, 
                            theta_sampler, 
                            dim, 
                            N_simulations, 
                            y_star, 
                            simulator_vectorized=simulator_vectorized)
        reference_table_theta[:, :, m_iteration] = prior_values
        reference_table_distances[:, :, m_iteration] = distances
    return(reference_table_theta, reference_table_distances)

def extract_mean_reference_table(reference_table_theta, reference_table_distances, threshold, target_function):
    """
    a function that extracts the needed information from the table
    """
    dim, N_simulations, M_repetitions = reference_table_theta.shape
    results_list  = []
    for m_iteration in range(M_repetitions):
        #pdb.set_trace()
        selector = reference_table_distances[:, :, m_iteration]<threshold
        selected_thetas = reference_table_theta[:, selector.flatten(), m_iteration]
        results_list.append(target_function(selected_thetas))
    return(results_list)

def target_function_mean(x):
    return(x.mean(axis=1).sum())

def target_function_var(x):
    return(x.var(axis=1).sum())

def loop_extraction_reference_talbe_aggregated(reference_table_theta, reference_table_distances, threshold_list, target_function):
    variance_results = np.zeros(len(threshold_list))
    iteration = 0
    for threshold in threshold_list:
        results_inter = extract_mean_reference_table(reference_table_theta, reference_table_distances, threshold, target_function)
        variance_results[iteration] = np.array(results_inter).var()
        iteration +=1
    return(variance_results)

def loop_extraction_reference_talbe_single(reference_table_theta, reference_table_distances, threshold_single, target_function):
    results_inter = extract_mean_reference_table(reference_table_theta, reference_table_distances, threshold_single, target_function)
    return(results_inter)


class compare_sampling_methods(object):
    """
    a class that allows the simulation for the different simulators
    """
    def __init__(self, 
        M_repetitions,
        simulator,
        delta,
        dim, 
        N_simulations,
        y_star,
        simulator_vectorized=True):

        self.M_repetitions = M_repetitions
        self.simulator = simulator
        self.delta = delta
        self.dim = dim
        self.N_simulations = N_simulations
        self.y_star = y_star
        self.simulator_vectorized = simulator_vectorized

    def generate_samples(self, theta_sampler, sampler_type):
        """
        help function that generates the samples
        """
        print("start simulation "+sampler_type+" dim "+str(self.dim))
        reference_table_theta, reference_table_distances = repeat_joint_simulation(
            self.M_repetitions,
            self.simulator,
            self.delta,
            theta_sampler,
            self.dim,
            self.N_simulations,
            self.y_star,
            simulator_vectorized=self.simulator_vectorized)

        if sampler_type == 'MC':
            self.reference_table_theta_mc = copy.copy(reference_table_theta)
            self.reference_table_distances_mc = copy.copy(reference_table_distances)

        elif sampler_type == 'QMC':
            self.reference_table_theta_qmc = copy.copy(reference_table_theta)
            self.reference_table_distances_qmc = copy.copy(reference_table_distances)


        elif sampler_type == 'RQMC':
            self.reference_table_theta_rqmc = copy.copy(reference_table_theta)
            self.reference_table_distances_rqmc = copy.copy(reference_table_distances)

        else: 
            raise ValueError('type of sampler does not exit !')
        

    def extract_information_aggregated_variance(self, threshold_quantiles, target_function, sampler_type):
        """
        function that applies the extraction to the simulated results
        """

        if sampler_type == 'MC':
            self.threshold_list = np.percentile(self.reference_table_distances_mc[0, :, 0], threshold_quantiles)
            print("extract variance MC")
            self.variance_results_mc = loop_extraction_reference_talbe_aggregated(
                self.reference_table_theta_mc,
                self.reference_table_distances_mc,
                self.threshold_list,
                target_function)

        elif sampler_type == 'QMC':
            print("extract variance QMC")
            self.variance_results_qmc = loop_extraction_reference_talbe_aggregated(
                self.reference_table_theta_qmc,
                self.reference_table_distances_qmc,
                self.threshold_list,
                target_function)

        elif sampler_type == 'RQMC':
            print("extract variance RQMC")
            self.variance_results_rqmc = loop_extraction_reference_talbe_aggregated(
                self.reference_table_theta_rqmc,
                self.reference_table_distances_rqmc,
                self.threshold_list,
                target_function)


    def extract_information_distribution(self, quantile_single, target_function, sampler_type):
        """
        function that extracts the distribution for a given quantile
        """
        if sampler_type == 'MC':
            self.threshold_single = np.percentile(self.reference_table_distances_mc[0, :, 0], quantile_single)
            print("extract variance MC")
            self.distribution_results_mc = loop_extraction_reference_talbe_single(
                self.reference_table_theta_mc, 
                self.reference_table_distances_mc, 
                self.threshold_single, 
                target_function)
            del self.reference_table_theta_mc
            del self.reference_table_distances_mc

        elif sampler_type == 'QMC':
            print("extract variance QMC")
            self.distribution_results_qmc = loop_extraction_reference_talbe_single(
                self.reference_table_theta_qmc, 
                self.reference_table_distances_qmc, 
                self.threshold_single, 
                target_function)
            del self.reference_table_theta_qmc
            del self.reference_table_distances_qmc

        elif sampler_type == 'RQMC':
            print("extract variance RQMC")
            self.distribution_results_rqmc = loop_extraction_reference_talbe_single(
                self.reference_table_theta_rqmc, 
                self.reference_table_distances_rqmc, 
                self.threshold_single,
                target_function)
            del self.reference_table_theta_rqmc
            del self.reference_table_distances_rqmc

    
    def simulate_and_extract(self, threshold_list, quantile_single, target_function, theta_sampler_list, sampler_type_list):
        """
        function that iterates the simulations and extracts the information
        """
        n_samplers = len(theta_sampler_list)
        for i_sampler in range(n_samplers):
            theta_sampler = theta_sampler_list[i_sampler]
            sampler_type = sampler_type_list[i_sampler]
            self.generate_samples(theta_sampler, sampler_type)
            self.extract_information_aggregated_variance(threshold_quantiles, target_function, sampler_type)
            self.extract_information_distribution(quantile_single, target_function, sampler_type)


def plot_variance_mean_variance(threshold_quantiles, instance_compare_samplers, name_plot):
    """
    a function that plots the variance reduction and that saves the figure
    """
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.plot(threshold_quantiles, instance_compare_samplers.variance_results_mc, label='MC', linewidth=3, linestyle='dashed')
    plt.plot(threshold_quantiles, instance_compare_samplers.variance_results_qmc, label='QMC', linewidth=3, linestyle='dotted')
    plt.plot(threshold_quantiles, instance_compare_samplers.variance_results_rqmc, label='RQMC', linewidth=3)
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Variance of the estimator', fontsize='14')
    plt.xlabel('Quantile of distance to y* in percent', fontsize='14')
    plt.legend(fontsize='14')
    plt.savefig(name_plot)
    plt.clf()

def plot_violin_plot(dict_dimensions_mc, dict_dimensions_qmc, dict_dimensions_rqmc, type_col):
    """
    a function that makes a violin plot
    """
    dimensions = dict_dimensions_mc.keys()
    results_list = []
    for dim in dimensions:
        mc_results = pd.DataFrame(dict_dimensions_mc[dim], columns=[type_col])
        mc_results['type'] = 'mc'
        qmc_results = pd.DataFrame(dict_dimensions_qmc[dim], columns=[type_col])
        qmc_results['type'] = 'qmc'
        rqmc_results = pd.DataFrame(dict_dimensions_rqmc[dim], columns=[type_col])
        rqmc_results['type'] = 'rqmc'

        frames = [mc_results, qmc_results, rqmc_results]
        result = pd.concat(frames)
        result['dim'] = dim
        results_list.append(result)
    #pdb.set_trace()
    results_total = pd.concat(results_list)
    if type_col == 'var':
        results_total['log_var'] = np.log(results_total['var'])
        sns.violinplot(x="dim", y='log_var', hue="type", data=results_total, palette="muted")
    else:
        sns.violinplot(x="dim", y=type_col, hue="type", data=results_total, palette="muted")
    plt.savefig(type_col+"violinplot_of_estimator_several_dim"+".png")
    plt.show()