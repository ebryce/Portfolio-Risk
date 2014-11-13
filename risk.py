import math
import statistics

global Rf

def sharpe(R):
    # Assumes a normal distribution of returns on security
    R_avg = mean(R)
    R_var = variance(R)
    S = (R_avg - Rf) / R_var
    return S

def beta(R, M):
    return (cov(R,M) / variance(M))

def C_derivative(time_to_maturity, strike, spot, Rf, volatility):
    d_a = (1 / (volatility * Math.sqrt(time_to_maturity))) * (Math.ln(strike / spot) + 0.5 * (volatility ** 2) * time_to_maturity)
    d_b = (1 / (volatility * Math.sqrt(time_to_maturity))) * (Math.ln(strike / spot) - 0.5 * (volatility ** 2) * time_to_maturity) + volatility * Math.sqrt(time_to_maturity)
    C = math.exp(-Rf * time_to_maturity) * (  )
    

def cov_array(R):
    ''' Return a covariance array on a set of securities with returns R
              R = [ [returns on security a], [returns on security b], ... , [returns on security n]
              becomes,
              [[   1    , cov(a,b), ... cov(a,n) ],
               [cov(b,a),    1    , ... cov(b,n) ],
                  ...
               [cov(n,a), cov(n,b), ...    1     ]]
    '''
    
    # since cov(a,b) = cov(b,a), we only need to calculate one way
    #     given cov(x,y) we need only do a calculation where x < y
    dim = len(R)
    cov_array = []
    for y in range(dim):
        # Note that nothing on the last row actually needs to be calculated
        row_cov_array = []
        for x in range(dim):
            if(x==y):
                # Covariance between two identical returns will always be 1.00
                this_cov = 1
            else if(x < y):
                # If this covariance has already been calculated, insert what 
                # is essential an element from the transposed matrix
                # Note that normally this array is in the form cov_array[y][x]
                this_cov = cov_array[x][y]
            else:
                # If this covariance has actually never been calculated before
                this_cov = cov(R[x], R[y])
            
            # Add covariances from this security (i.e. this y,x) to 1-D array
            row_cov_array.append(this_cov)
        # Add all covariances from this row (i.e. this y) to the 2-D array
        cov_array.append(row_cov_array)
    return cov_array
            
        

def cov(Ra, Rb):
    ''' Return the coveriance between two securites with returns Ra and Rb
            Ra = [return on security a]
            Rb = [return on security b]'''
    if(len(Ra) > len(Rb)): N = len(Rb)
    else: N = len(Ra)
    
    Ra_avg = mean(Ra)
    Rb_avg = mean(Rb)
    
    S = 0
    for i in range(N):
        term = (Ra[i] - Ra_avg) * (Rb[i] - Rb_avg)
        S += term
    S = S / (n - 1)
    return S