from matplotlib import pyplot as plt
import pickle

out_rqmc = pickle.load( open( "tuber_rqmc_simulation_abc_epsilon_0.2.p", "rb" ) )
out_mc = pickle.load( open( "tuber_mc_simulation_abc_epsilon_0.2.p", "rb" ) )

var_list_rqmc = out_rqmc[3]
var_list_mc = out_mc[3]
var_rqmc = [i[2,2] for i in var_list_rqmc]

var_mc = [i[2,2] for i in var_list_mc]

plt.plot([50,100,200,500,1000], var_rqmc, label = "RQMC")
plt.plot([50,100,200,500,1000], var_mc, label = "MC")
plt.ylabel('Standard deviation of estimated mean')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('logscale')
plt.ylabel('logscale')
plt.title('Standard deviation estimated mean qm and rqmc epsilon = 0.01')
plt.grid(True)
plt.legend()
plt.show()


out_rqmc[2][4][:2,]
plt.plot(out_rqmc[2][4][2,], out_rqmc[2][4][1,], 'ro')
plt.show()
