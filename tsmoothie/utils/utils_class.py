'''
A collection of utility classes.
'''

import numpy as np
from scipy import sparse



class LinearRegression(object):
    
    """
    Ordinary least squares Linear Regression. 
    
    Linear model that estimates coefficients to minimize the residual 
    sum of squares between the observed targets and the predictions. 
    It automatically handles single and multiple targets. 
    
    It's a modified version of the Linear Regression implemented in scikit-learn. 
    
    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to calculate the intercept for this model. If set 
        to False, no intercept will be used in calculations.
        
    Attributes
    ----------
    coef_ : array of shape (n_coef,) for univariate data or 
        (sample, n_coef) for multivariate data 
        Array containing the estimated coefficients of the linear model. 
        Available after fitting. 
    residues_ : array of shape (sample,) 
        Sums of residuals; squared Euclidean 2-norm for each sample. 
        Available after fitting. 
    rank_ : int 
        Rank of exogenous variable matrix. 
        Available after fitting. 
    singular_ : array of shape (n_coef,) 
        Singular values of the exogenous variable matrix. 
        Available after fitting. 
    intercept_ : array of shape (sample,) 
        Array containing the estimated intercepts of the linear model. 
        If fit_intercept = False it returns 0. 
        Available after fitting.
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        
        
    def __repr__(self):
        return "<tsmootie.utils.utils_class.LinearRegression>"
    
    
    def __str__(self):
        return "<tsmootie.utils.utils_class.LinearRegression>"
        

    def _preprocess_data(self, X, y, sample_weight):
        
        """
        Center and scale data. 
        Centers data to have mean zero along axis 0. If fit_intercept=False 
        no centering is done. If sample_weight is not None, then the weighted mean 
        of X and y is zero, and not the mean itself. 
        This is here because nearly all linear models will want their data to be 
        centered. This function also systematically makes y consistent with X.dtype.
        
        Returns
        -------
        X : array 
        y : array 
        X_offset : array 
        y_offset : array 
        X_scale : array 
        """
        
        X = X.copy(order='K')
        y = y.copy(order='K')
        y = np.asarray(y, dtype=X.dtype)

        if self.fit_intercept:

            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

            y_offset = np.average(y, axis=0, weights=sample_weight)
            y -= y_offset

        else:

            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

            if y.ndim == 1:
                y_offset = X.dtype.type(0)
            else:
                y_offset = np.zeros(y.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale


    def _rescale_data(self, X, y, sample_weight):
        
        """
        Rescale data sample-wise by square root of sample_weight. 

        Returns
        -------
        X_rescaled : array 
        y_rescaled : array
        """

        n_samples = X.shape[0]
        sample_weight = np.sqrt(sample_weight)
        sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                      shape=(n_samples, n_samples))
        X = sw_matrix @ X
        y = sw_matrix @ y

        return X, y
    

    def fit(self, X, y, sample_weight):
        
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features) 
            Training data. 
        y : array of shape (n_samples,) or (n_samples, n_targets) 
            Target values. 
        sample_weight : array of shape (n_samples,), default None 
            Individual weights for each sample. 
        
        Returns
        -------
        self : returns an instance of self 
        """

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, 
            sample_weight=sample_weight)

        if np.unique(sample_weight).shape[0] > 1:
            X, y = self._rescale_data(X, y, sample_weight)

        self.coef_, self.residues_, self.rank_, self.singular_ = np.linalg.lstsq(X, y, rcond=None)

        self.coef_ = self.coef_.T
        
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.
            
        return self
                    
                
    def predict(self, X):
        
        """
        Compute the predictions with fitted coefficients. 
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features) 
            Exogenous data.
        
        Returns
        -------
        pred : array
        """
        
        pred = (X @ self.coef_.T) + self.intercept_
        
        return pred