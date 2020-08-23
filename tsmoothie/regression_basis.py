'''
Basis functions for regression.
Inspired by: https://github.com/madrury/basis-expansions/blob/master/examples/comparison-of-smoothing-methods.ipynb
'''

import numpy as np



def polynomial(degree, basis_len):
    
    """
    Create basis for polynomial regression.

    Returns
    -------
    X_base : array
        Basis for polynomial regression. 
    """
    
    X = np.arange(basis_len, dtype=np.float64)
    X_base = np.repeat([X], degree, axis=0).T
    X_base = np.power(X_base, np.arange(1,degree+1))
    
    return X_base



def linear_spline(knots, basis_len):
    
    """
    Create basis for linear spline regression. 

    Returns
    -------
    X_base : array 
        Basis for linear spline regression. 
    """
    
    n_knots = len(knots)
    X = np.arange(basis_len)
    
    X_base = np.zeros((basis_len, n_knots + 1))
    X_base[:, 0] = X

    X_base[:,1:] = X[:,None] - knots[None,:]
    X_base[X_base<0] = 0
    
    return X_base



def cubic_spline(knots, basis_len):
    
    """
    Create basis for cubic spline regression. 

    Returns
    -------
    X_base : array 
        Basis for cubic spline regression. 
    """
    
    n_knots = len(knots)
    X = np.arange(basis_len)
    
    X_base = np.zeros((basis_len, n_knots + 3))
    X_base[:, 0] = X
    X_base[:, 1] = X_base[:, 0] * X_base[:, 0]
    X_base[:, 2] = X_base[:, 1] * X_base[:, 0]

    X_base[:,3:] = np.power(X[:,None] - knots[None,:], 3)
    X_base[X_base<0] = 0
    
    return X_base



def natural_cubic_spline(knots, basis_len):
    
    """
    Create basis for natural cubic spline regression. 

    Returns
    -------
    X_base : array 
        Basis for natural cubic spline regression. 
    """
    
    n_knots = len(knots)
    X = np.arange(basis_len)

    X_base = np.zeros((basis_len, n_knots - 1))
    X_base[:, 0] = X

    numerator1 = X[:,None] - knots[None, :n_knots - 2]
    numerator1[numerator1<0] = 0
    numerator2 = X[:,None] - knots[None, n_knots - 1]
    numerator2[numerator2<0] = 0

    numerator = np.power(numerator1, 3) - np.power(numerator2, 3)
    denominator = knots[n_knots - 1] - knots[:n_knots - 2]

    numerator1_dd = X[:,None] - knots[None, n_knots - 2]
    numerator1_dd[numerator1_dd<0] = 0
    numerator2_dd = X[:,None] - knots[None, n_knots - 1]
    numerator2_dd[numerator2_dd<0] = 0

    numerator_dd = np.power(numerator1_dd, 3) - np.power(numerator2_dd, 3)
    denominator_dd = knots[n_knots - 1] - knots[n_knots - 2]

    dd = numerator_dd / denominator_dd
    
    X_base[:, 1:] = numerator / denominator - dd
    
    return X_base



def gaussian_kernel(knots, sigma, basis_len):
    
    """
    Create basis for gaussian kernel regression. 

    Returns
    -------
    X_base : array 
        Basis for gaussian kernel regression. 
    """
    
    n_knots = len(knots)
    X = np.arange(basis_len) / basis_len
    
    X_base = - np.square(X[:,None] - knots) / (2 * sigma)
    X_base = np.exp(X_base)
    
    return X_base



def binner(knots, basis_len):
    
    """
    Create basis for binner regression. 

    Returns
    -------
    X_base : array 
        Basis for binner regression. 
    """
    
    n_knots = len(knots)
    X = np.arange(basis_len)
    
    X_base = np.zeros((basis_len, n_knots + 1))
    X_base[:, 0] = X <= knots[0]

    X_base[:,1:-1] = np.logical_and(X[:,None] <= knots[1:][None,:], 
                                    X[:,None] > knots[:(n_knots - 1)][None,:])

    X_base[:, n_knots] = knots[-1] < X
    
    return X_base



def lowess(smooth_fraction, basis_len):
    
    """
    Create basis for LOWESS. 

    Returns
    -------
    X_base : array 
        Basis for LOWESS. 
    """
    
    X = np.arange(basis_len)
    
    r = int(np.ceil(smooth_fraction * basis_len))
    r = min(r, basis_len-1)
    
    X = X[:,None] - X[None,:]
    
    h = np.sort(np.abs(X), axis=1)[:,r]
    
    X_base = np.abs(X / h).clip(0.0, 1.0)
    X_base = np.power(1 - np.power(X_base, 3), 3)
    
    return X_base