require(smfsb)
data(LVdata)
library(timeit)

simTs(c(50,100),0,30,2,stepLVc)

N=1e5
message(paste("N =",N))
prior=cbind(th1=exp(runif(N,-6,2)),th2=exp(runif(N,-6,2)),th3=exp(runif(N,-6,2)))
rows=lapply(1:N,function(i){prior[i,]})
message("starting simulation")
samples=lapply(rows,function(th){simTs(c(50,100),0,30,2,stepLVc,th)})
message("finished simulation")

distance<-function(ts)
{
  diff=ts-LVperfect
  sum(diff*diff)
}

message("computing distances")
dist=lapply(samples,distance)
message("distances computed")

dist=sapply(dist,c)
cutoff=quantile(dist,1000/N)
post=prior[dist<cutoff,]

op=par(mfrow=c(2,3))
apply(post,2,hist,30)
apply(log(post),2,hist,30)
par(op)
