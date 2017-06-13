import numpy as np
import ot

n = 1000
x = np.random.normal(loc=-1, size = (n,2))
y = np.random.normal(loc=1, size = (n,2))
M = ot.dist(x,y)
a = ot.unif(n)
b = ot.unif(n)
ot.emd(a, b, M)
ot.lp.emd(a, b, M)
ot.sinkhorn(a,b,M,1)

