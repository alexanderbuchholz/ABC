# simulator r_based

require(smfsb)
data(LVdata)


robservation <- function(theta){
    return(simTs(c(50,100),0,30,2,stepLVc, theta))
}

y_star <- function(){
    return(LVperfect)
}