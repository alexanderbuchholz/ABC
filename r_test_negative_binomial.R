# negative binomial distribution 
n = 100000
size = 2
t = exp((1:180/10))
probs = matrix(1/t)

neg_binomial = function(p) rnbinom(n, size, p)
res = apply(probs, 1, neg_binomial)
means_tries = apply(res, 2, mean)
vars_tries = apply(res, 2, var)
quantile_99_percent = function(x) quantile(x, 0.999)
quantile_tries = apply(res, 2, quantile_99_percent)
max_tries = apply(res, 2, max)
median_tries = apply(res, 2, median)


plot(probs, means_tries, log='xy')
lines(probs, vars_tries**0.5)
lines(probs, quantile_tries)
lines(probs, max_tries)
lines(probs, median_tries)

hist(res[,100])
probs[100]
