## functions for computing expected off- / on-target assocation rates 

off_target <- function(Vgt,          ## true polygenic score covariance matrix
                       Vg0,          ## random mating pgs covariance matrix
                       Ve,           ## residual covariance matrix
                       Mk,           ## causal variants per phenotype
                       NN,           ## GWAS sample size
                       alpha,        ## GWAS significance threshold
                       NMC = 2000) { ## number of samples for monte carlo integration
    nrep <- max(1,floor(NMC / Mk))
    mean(replicate(nrep, {
    K <- dim(Vgt)[1]
    NMC <- min(NMC, Mk)
    B <- construct_betas(Mk=Mk, Vg0=Vg0, K=K)
    subinds <- sort(sample((Mk+1):(Mk*K), NMC))
    A = solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    Vy = Vgt + Ve
    A = solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    lambda_null <- sapply(subinds, 
                          function(j)  {
                              b1 <- B[j,,drop=F]
                              beta <- B[j,1]
                              sx2 <- 1 + b1 %*% A %*% t(b1)
                              bias <- b1 %*% A %*% t(B[-j,]) %*% B[-j,1]
                              Vresid <- Vy[1,1]-beta^2 * sx2
                              NN * sx2 * (beta + bias)^2 / Vresid
                              }
                         )
    mean(chi2_upper_tail(alpha, lambda_null))
                         }))}


on_target <- function(Vgt,          ## true polygenic score covariance matrix
                      Vg0,          ## random mating pgs covariance matrix
                      Ve,           ## residual covariance matrix
                      Mk,           ## causal variants per phenotype
                      NN,           ## GWAS sample size
                      alpha,        ## GWAS significance threshold
                      NMC = 2000) { ## number of samples for monte carlo integration
    nrep <- max(1,floor(NMC / Mk))
    mean(replicate(nrep, {
    K <- dim(Vgt)[1]
    NMC <- min(NMC, Mk)
    B <- construct_betas(Mk=Mk, Vg0=Vg0, K=K)
    subinds <- sort(sample(1:Mk, NMC))
    A = solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    Vy = Vgt + Ve
    A = solve(Vg0, Vgt) %*% solve(Vg0) - solve(Vg0)
    lambda_true <- sapply(subinds, 
                          function(j)  {
                              b1 <- B[j,,drop=F]
                              beta <- B[j,1]
                              sx2 <- 1 + b1 %*% A %*% t(b1)
                              bias <- b1 %*% A %*% t(B[-j,]) %*% B[-j,1]
                              Vresid <- Vy[1,1]-beta^2 * sx2
                              NN * sx2 * (beta + bias)^2 / Vresid
                              })
    
    mean(chi2_upper_tail(alpha, lambda_true))
                         }))}