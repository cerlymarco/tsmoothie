'''
Define Bootstrapping class.
'''

import numpy as np

from .utils_func import _id_nb_bootstrap, _id_mb_bootstrap, _id_cb_bootstrap, _id_s_bootstrap



class BootstrappingWrapper(object):

    """
    BootstrappingWrapper generates new timeseries samples using specific 
    algorithms for sequences bootstrapping.
    
    The BootstrappingWrapper handles single timeseries. Firstly, the smoothing  
    of the received series is computed. Secondly, the residuals of the smoothing 
    operation are generated and randomly partitioned into blocks according to the
    chosen bootstrapping techniques. Finally, the residual blocks are sampled in
    random order, concatenated and then added to the original smoothing curve in
    order to obtain a bootstrapped timeseries.
    
    The supported bootstrap algorithms are:
     - none overlapping block bootstrap ('nbb')
     - moving block bootstrap ('mbb')
     - circular block bootstrap ('cbb')
     - stationary bootstrap ('sb')
    
    Parameters
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother (except for WindowWrapper). 
        It computes the smoothing on the series received.        
    bootstrap_type : str
        The type of algorithm used to compute the bootstrap. 
        Supported types are: none overlapping block bootstrap ('nbb'), 
        moving block bootstrap ('mbb'), circular block bootstrap ('cbb'), 
        stationary bootstrap ('sb').
    block_length : int 
        The shape of the blocks used to sample from the residuals of the 
        smoothing operation and used to bootstrap new samples. 
        Must be an integer in [3, timesteps).
        
    Attributes
    ----------   
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother (except for WindowWrapper)
        that was passed to BootstrappingWrapper. It as the same properties and 
        attributes of every Smoother. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_seasonal_data
    >>> from tsmoothie.bootstrap import BootstrappingWrapper
    >>> from tsmoothie.smoother import *
    >>> np.random.seed(33)
    >>> data = sim_seasonal_data(n_series=1, timesteps=200, 
    ...                          freq=24, measure_noise=10)
    >>> bts = BootstrappingWrapper(ConvolutionSmoother(window_len=8, window_type='ones'), 
    ...                            bootstrap_type='mbb', block_length=24)
    >>> bts_samples = bts.sample(data, n_samples=100)
    """

    def __init__(self, Smoother, bootstrap_type, block_length):
        self.Smoother = Smoother
        self.bootstrap_type = bootstrap_type
        self.block_length = block_length
        self.__name__ = 'tsmoothie.bootstrap.BootstrappingWrapper'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def sample(self, data, n_samples=1):

        """
        Bootstrap timeseries.
        
        Parameters
        ----------
        data : array-like of shape (1, timesteps) or also (timesteps,) 
            Single timeseries to bootstrap. The data are assumed to be in increasing time order. 
        n_samples : int, default 1
            How many bootstrapped series to generate.
               
        Returns
        -------
        bootstrap_data : array of shape (n_samples, timesteps)
            Bootstrapped samples.
        """
        
        bootstrap_types = ['nbb', 'mbb', 'cbb', 'sb']

        if self.bootstrap_type not in bootstrap_types:
            raise ValueError(f"'{self.bootstrap_type}' is not a supported bootstrap type. "
                             f"Supported types are {bootstrap_types}")

        if not hasattr(self.Smoother, '__name__'):
            raise ValueError("Use a Smoother from tsmoothie.smoother")

        if 'tsmoothie.smoother' not in self.Smoother.__name__:
            raise ValueError("Use a Smoother from tsmoothie.smoother")
            
        if self.Smoother.__name__ == 'tsmoothie.smoother.WindowWrapper':
            raise ValueError("WindowWrapper doesn't not support bootstrapping")
            
        if self.block_length < 3:
            raise ValueError("block_length must be >= 3")
            
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        data = np.asarray(data)
        if np.prod(data.shape) == np.max(data.shape):
            data = data.ravel()
            nobs = data.shape[0]
            if self.block_length >= nobs:
                raise ValueError("block_length must be < than the timesteps dimension of the data passed")
            if self.Smoother.__name__ == 'tsmoothie.smoother.ExponentialSmoother':
                nobs = data.shape[0] - self.Smoother.window_len
                if self.block_length >= nobs:
                    raise ValueError("block_length must be < than (timesteps - window_len)")
        else:
            raise ValueError("The format of data received is not appropriate. " 
                             "BootstrappingWrapper accepts only univariate timeseries")
        
        self.Smoother.copy = True
        self.Smoother.smooth(data)
        residuals = self.Smoother.data - self.Smoother.smooth_data
        residuals = np.nan_to_num(residuals, nan=0)
        
        if self.bootstrap_type == 'nbb':
            bootstrap_func = _id_nb_bootstrap
        elif self.bootstrap_type == 'mbb':
            bootstrap_func = _id_mb_bootstrap
        elif self.bootstrap_type == 'cbb':
            bootstrap_func = _id_cb_bootstrap
        else: 
            bootstrap_func = _id_s_bootstrap
        
        bootstrap_data = np.empty((n_samples, nobs))
        for i in np.arange(n_samples):
            bootstrap_id = bootstrap_func(nobs, self.block_length)
            bootstrap_res = residuals[[0], bootstrap_id]
            bootstrap_data[i] = self.Smoother.smooth_data + bootstrap_res
        
        return bootstrap_data