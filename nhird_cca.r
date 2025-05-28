r_mate <- read.csv('nhird_cors.csv')

f_ind <- (grep('^f', names(r_mate)))
m_ind <- (grep('^m', names(r_mate)))

CXX <- as.matrix(r_mate[f_ind, f_ind])
CXY <- as.matrix(r_mate[f_ind, m_ind])
CYY <- as.matrix(r_mate[m_ind, m_ind])
CYX <- as.matrix(r_mate[m_ind, f_ind])
ev <- eigen(solve(CXX, CXY %*% solve(CYY, CYX)))
D <- diag(ev$values)
Dinv <- diag(1/diag(D))
Qx <- ev$vectors
Qy = solve(CYY, CYX %*% Qx %*% Dinv)

cancors <- sqrt(D)
# cancors
Sx = t(Qx) %*% CXX %*% Qx
Sy = t(Qy) %*% CYY %*% Qy

## Canonical coefficients
Wx = Qx %*% diag(1/sqrt(diag(Sx)))
Wy = Qy %*% diag(1/sqrt(diag(Sy)))

## Structure matrices (correlations between variables and canonical variates)
Rx = CXX %*% Wx  # Correlations between X variables and their canonical variates
Ry = CYY %*% Wy  # Correlations between Y variables and their canonical variates

# Compute squared structure coefficients
Rx2 = Rx^2
Ry2 = Ry^2


# Average variance explained in each set by their own canonical variates
varexp_x = colMeans(Rx2)  # Average variance explained in X by each X canonical variate
varexp_y = colMeans(Ry2)  # Average variance explained in Y by each Y canonical variate

# Redundancy indices
# How much variance in Y is explained by X canonical variates
red_y_x = diag(varexp_y * cancors^2)

# How much variance in X is explained by Y canonical variates
red_x_y = diag(varexp_x * cancors^2)

# Total redundancy
total_red_y_x = sum(red_y_x)  # Total redundancy of Y given X
total_red_x_y = sum(red_x_y)  # Total redundancy of X given Y

## mean red = redundancy
tmp <- .5*((red_y_x)/total_red_y_x + (red_x_y)/total_red_x_y)
mean_red <- .5*(cumsum(red_y_x[order(tmp,decreasing = T)])/total_red_y_x + cumsum(red_x_y[order(tmp,decreasing = T)])/total_red_x_y)
