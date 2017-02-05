import numpy.random as nr
import numpy as np
import cProfile

theta = np.array([[0.4],[0.3],[0.3]])
def multi_sampling(theta):
	return nr.multinomial(1,theta[:,0],1)==1

def choice_sampling(theta):
	n = len(theta)
	return nr.choice(range(n),p=theta[:,0],size=1)

def repeat_fun(func,N,arg):
	for i in range(N):
		func(arg)
print multi_sampling(theta)
print choice_sampling(theta)
N = 100000
cProfile.run('repeat_fun(multi_sampling,N,theta)')
cProfile.run('repeat_fun(choice_sampling,N,theta)')

