'''
Define Smoother classes.
'''

import numpy as np
from scipy.signal import fftconvolve
import simdkalman

from .utils_class import LinearRegression
from .utils_func import (create_windows, sigma_interval, kalman_interval,
                         confidence_interval, prediction_interval)
from .utils_func import (_check_noise_dict, _check_knots, _check_weights,
                         _check_data, _check_data_nan, _check_output)
from .regression_basis import (polynomial, linear_spline, cubic_spline, natural_cubic_spline,
                               gaussian_kernel, binner, lowess)



class ExponentialSmoother(object):

    """
    ExponentialSmoother operates convolutions of fixed dimensions 
    on the series using a weighted windows. The weights are the same 
    for all windows and are computed using an exponential decay. 
    The most recent observations are most important than the past ones. 
    This is imposed choosing a parameter (alpha). 
    No padded is provided in order to not alter the results at the edges. 
    For this reason, this technique doesn't operate smoothing until 
    the observations at position window_len.
    
    The ExponentialSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    window_len : int
        Greater than equal to 1. The length of the window used to compute 
        the exponential smoothing.
    alpha : float
        Between 0 and 1. (1-alpha) provides the importance of the past obsevations 
        when computing the smoothing.
    copy : bool, default True
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------   
    smooth_data : array of shape (series, timesteps-window_len)
        Smoothed data derived from the smoothing operation. 
        It has the same shape of the raw data received without the first observations 
        until window_len. It is accessible after computhing smoothing, 
        otherwise None is returned. 
    data : array of shape (series, timesteps-window_len) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = ExponentialSmoother(window_len=20, alpha=0.3)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, window_len, alpha, copy=True):
        self.window_len = window_len
        self.alpha = alpha
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.__name__ = 'tsmoothie.smoother.ExponentialSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps-window_len) or also 
            (timesteps-window_len,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
               
        Returns
        -------
        self : returns an instance of self
        """

        if self.window_len < 1:
            raise ValueError("window_len must be >= 1")

        if self.alpha > 1 or self.alpha < 0:
            raise ValueError("alpha must be in the range [0,1]")

        data = _check_data(data)

        if self.window_len >= data.shape[0]:
            raise ValueError("window_len must be < than timesteps dimension of the data received")

        w = np.power((1-self.alpha), np.arange(self.window_len))

        if data.ndim == 2:
            w = np.repeat([w/w.sum()], data.shape[1], axis=0).T
        else:
            w = w/w.sum()

        smooth = fftconvolve(w, data, mode='full', axes=0)
        smooth = smooth[self.window_len:data.shape[0]]
        data = data[self.window_len:data.shape[0]]

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.data = data

        return self


    def get_intervals(self, interval_type, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported type is 'sigma_interval'. 
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps-window_len) 
            Lower bands. 
        up : array of shape (series, timesteps-window_len) 
            Upper bands. 
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)

        return low, up



class ConvolutionSmoother(object):

    """
    ConvolutionSmoother operates convolutions of fixed dimensions 
    on the series using a weighted windows. The weights can assume 
    different format but they are the same for all the windows and 
    fixed for the whole procedure. The series are padded, reflecting themself, 
    with a quantity equal to the window size in both ends to avoid loss of information.
    
    The ConvolutionSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    window_len : int
        Greater than equal to 1. The length of the window used to compute the convolutions.
    window_type : str
        The type of the window used to compute the convolutions. 
        Supported types are: 'ones', 'hanning', 'hamming', 'bartlett', 'blackman'.
    copy : bool, default True
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = ConvolutionSmoother(window_len=10, window_type='ones')
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('sigma_interval')
    """

    def __init__(self, window_len, window_type, copy=True):
        self.window_len = window_len
        self.window_type = window_type
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.__name__ = 'tsmoothie.smoother.ConvolutionSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
               
        Returns
        -------
        self : returns an instance of self
        """

        window_types = ['ones', 'hanning', 'hamming', 'bartlett', 'blackman']

        if self.window_type not in window_types:
            raise ValueError(f"'{self.window_type}' is not a supported window type. Supported types are {window_types}")

        if self.window_len < 1:
            raise ValueError("window_len must be >= 1")

        data = _check_data(data)

        if self.window_len % 2 == 0:
            self.window_len = int(self.window_len+1)

        if self.window_type == 'ones':
            w = np.ones(self.window_len)
        else:
            w = eval('np.'+self.window_type+'(window_len)')

        if data.ndim == 2:
            pad_data = np.pad(data, ((self.window_len,self.window_len),(0,0)), mode='symmetric')
            w = np.repeat([w/w.sum()], pad_data.shape[1], axis=0).T
        else:
            pad_data = np.pad(data, self.window_len, mode='symmetric')
            w = w/w.sum()

        smooth = fftconvolve(w, pad_data, mode='valid', axes=0)
        smooth = smooth[(self.window_len//2+1):-(self.window_len//2+1)]

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.data = data

        return self


    def get_intervals(self, interval_type, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported type is 'sigma_interval'. 
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands.
        up : array of shape (series, timesteps) 
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)

        return low, up



class PolynomialSmoother(object):

    """
    PolynomialSmoother smoothes the timeseries applying a linear regression 
    on an ad-hoc basis expansion. 
    The input space, used to build the basis expansion, consists in 
    a single continuos increasing sequence.
    
    The PolynomialSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    degree : int
        The polynomial order used to build the basis. 
    copy : bool, default True 
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = PolynomialSmoother(degree=6)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, degree, copy=True):
        self.degree = degree
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.X = None
        self.__name__ = 'tsmoothie.smoother.PolynomialSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data, weights=None):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
        weights : array-like of shape (timesteps,), default None 
            Individual weights for each timestep. In case of multidimesional timeseries,  
            the same weights are used for all the timeseries. 
               
        Returns
        -------
        self : returns an instance of self
        """

        if self.degree < 1:
            raise ValueError("degree must be > 0")

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        X_base = polynomial(self.degree, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.X = X_base
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str 
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval', 'confidence_interval' and 'prediction_interval'. 
        confidence : float, default 0.05 
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored. 
        n_sigma : int, default 2 
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored. 
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands. 
        up : array of shape (series, timesteps) 
            Upper bands. 
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'confidence_interval', 'prediction_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'confidence_interval' or interval_type == 'prediction_interval':
            interval_f = eval(interval_type)
            low, up = interval_f(self.data, self.smooth_data, self.X, confidence)
            return low, up

        return low, up



class SplineSmoother(object):

    """
    SplineSmoother smoothes the timeseries applying a linear regression 
    on an ad-hoc basis expansion. Three types of spline smoothing are 
    available: 'linear spline', 'cubic spline', 'natural cubic spline'. 
    In all of the available methods, the input space consists in a single 
    continuos increasing sequence. 
    
    Two possibilities are available:
    - smooth the timeseries in equal intervals, where the number of intervals
      is a user defined parameter (n_knots); 
    - smooth the timeseries in custom length intervals, where the interval positions 
      are defined by the user as normalize points (knots). 
    The two methods are exclusive: the usage of n_knots makes not effective 
    the usage of knots and vice-versa.
    
    The SplineSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    spline_type : str
        Type of spline smoother to operate. Supported types are 'linear_spline', 
        'cubic_spline' or 'natural_cubic_spline' 
    n_knots : int
        Between 1 and timesteps for 'linear_spline' and 'natural_cubic_spline'. 
        Between 3 and timesteps for 'natural_cubic_spline'. 
        Number of equal intervals used to divide the input 
        space and smooth the timeseries. A lower value of n_knots 
        will result in a smoother curve 
    knots : array-like of shape (n_knots,), default None 
        With length of at least 1 for 'linear_spline' and 'natural_cubic_spline'. 
        With length of at least 3 for 'natural_cubic_spline'. 
        Normalized points in the range [0,1] that specify 
        in which sections divide the input space. A lower number of knots 
        will result in a smoother curve 
    copy : bool, default True 
        If True, the raw data received to the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = SplineSmoother(n_knots=6, spline_type='natural_cubic_spline')
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, spline_type, n_knots, knots=None, copy=True):
        self.spline_type = spline_type
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.X = None
        self.__name__ = 'tsmoothie.smoother.SplineSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data, weights=None):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries. 
        weights : array-like of shape (timesteps,), default None 
            Individual weights for each timestep. In case of multidimesional timeseries, 
            the same weights are used for all the timeseries. 
               
        Returns
        -------
        self : returns an instance of self
        """

        spline_types = {'linear_spline':1, 'cubic_spline':1, 'natural_cubic_spline':3}

        if self.spline_type not in spline_types:
            raise ValueError(f"'{self.spline_type}' is not a supported spline type. Supported types are {list(spline_types.keys())}")

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            self.knots = _check_knots(self.knots, spline_types[self.spline_type])[1:-1] * basis_len

        else:
            if self.n_knots < spline_types[self.spline_type]:
                raise ValueError(f"'{self.spline_type}' requires n_knots >= {spline_types[self.spline_type]}")

            if self.n_knots > basis_len:
                raise ValueError("n_knots must be <= than timesteps dimension of the data received")

            self.knots = np.linspace(0, basis_len, self.n_knots + 2)[1:-1]

        f = eval(self.spline_type)
        X_base = f(self.knots, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.X = X_base
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval', 'confidence_interval' and 'prediction_interval'. 
        confidence : float, default 0.05 
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored. 
        n_sigma : int, default 2 
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps)
            Lower bands.
        up : array of shape (series, timesteps)
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'confidence_interval', 'prediction_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'confidence_interval' or interval_type == 'prediction_interval':
            interval_f = eval(interval_type)
            low, up = interval_f(self.data, self.smooth_data, self.X, confidence)
            return low, up

        return low, up



class GaussianSmoother(object):

    """
    GaussianSmoother smoothes the timeseries applying a linear regression 
    on an ad-hoc basis expansion. The features created with this method 
    are obtained applying a gaussian kernel centered to specified points 
    of the input space. 
    In timeseries domain, the input space consists in a single continuos 
    increasing sequence.
    
    Two possibilities are available: 
    - smooth the timeseries in equal intervals, where the number of intervals 
      is a user defined parameter (n_knots); 
    - smooth the timeseries in custom length intervals, where the interval positions 
      are defined by the user as normalize points (knots). 
    The two methods are exclusive: the usage of n_knots makes not effective 
    the usage of knots and vice-versa.
    
    The GaussianSmoother automatically vectorizes, in an efficient way,  
    the desired smoothing operation on all the series received. 
    
    Parameters
    ----------
    n_knots : int
        Between 1 and timesteps. Number of equal intervals used to divide the input  
        space and smooth the timeseries. A lower value of n_knots 
        will result in a smoother curve. 
    knots : array-like of shape (n_knots,), default None 
        With length of at least 1. Normalized points in the range [0,1] that specify 
        in which sections divide the input space. A lower number of knots 
        will result in a smoother curve. 
    copy : bool, default True 
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = GaussianSmoother(n_knots=6, sigma=0.1)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, sigma, n_knots, knots=None, copy=True):
        self.sigma = sigma
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.X = None
        self.__name__ = 'tsmoothie.smoother.GaussianSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data, weights=None):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
        weights : array-like of shape (timesteps,), default None 
            Individual weights for each timestep. In case of multidimesional timeseries, 
            the same weights are used for all the timeseries.
               
        Returns
        -------
        self : returns an instance of self
        """

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            self.knots = _check_knots(self.knots, 1)[1:-1]

        else:
            if self.n_knots < 1:
                raise ValueError("n_knots must be > 0")

            if self.n_knots > basis_len:
                raise ValueError("n_knots must be <= than timesteps dimension of the data received")

            self.knots = np.linspace(0, 1, self.n_knots + 2)[1:-1]

        X_base = gaussian_kernel(self.knots, self.sigma, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.X = X_base
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str 
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval', 'confidence_interval' and 'prediction_interval'. 
        confidence : float, default 0.05 
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored. 
        n_sigma : int, default 2 
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands.
        up : array of shape (series, timesteps) 
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'confidence_interval', 'prediction_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'confidence_interval' or interval_type == 'prediction_interval':
            interval_f = eval(interval_type)
            low, up = interval_f(self.data, self.smooth_data, self.X, confidence)
            return low, up

        return low, up



class BinnerSmoother(object):

    """
    BinnerSmoother smoothes the timeseries applying a linear regression 
    on an ad-hoc basis expansion. The features created with this method 
    are obtained binning the input space into intervals. 
    An indicator feature is created for each bin, indicating where 
    a given observation falls into. 
    In timeseries domain, the input space consists in a single continuos 
    increasing sequence.
    
    Two possibilities are available:
    - smooth the timeseries in equal intervals, where the number of intervals 
      is a user defined parameter (n_knots); 
    - smooth the timeseries in custom length intervals, where the interval positions 
      are defined by the user as normalize points (knots). 
    The two methods are exclusive: the usage of n_knots makes not effective 
    the usage of knots and vice-versa.
    
    The BinnerSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    n_knots : int
        Between 1 and timesteps. Number of equal intervals used to divide the input 
        space and smooth the timeseries. A lower value of n_knots 
        will result in a smoother curve. 
    knots : array-like of shape (n_knots,), default None 
        With length of at least 1. Normalized points in the range [0,1] that specify 
        in which sections divide the input space. A lower number of knots 
        will result in a smoother curve.
    copy : bool, default True
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = BinnerSmoother(n_knots=6)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, n_knots, knots=None, copy=True):
        self.n_knots = n_knots
        self.knots = knots
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.X = None
        self.__name__ = 'tsmoothie.smoother.BinnerSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data, weights=None):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
        weights : array-like of shape (timesteps,), default None 
            Individual weights for each timestep. In case of multidimesional timeseries, 
            the same weights are used for all the timeseries.
               
        Returns
        -------
        self : returns an instance of self
        """

        data = _check_data(data)
        basis_len = data.shape[0]
        weights = _check_weights(weights, basis_len)

        if self.knots is not None:
            self.knots = _check_knots(self.knots, 1)[1:-1] * basis_len

        else:
            if self.n_knots < 1:
                raise ValueError("n_knots must be > 0")

            if self.n_knots > basis_len:
                raise ValueError("n_knots must be <= than timesteps dimension of the data received")

            self.knots = np.linspace(0, basis_len, self.n_knots + 2)[1:-1]

        X_base = binner(self.knots, basis_len)

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_base, data, sample_weight=weights)

        smooth = lr.predict(X_base)

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.X = X_base
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval', 'confidence_interval' and 'prediction_interval'. 
        confidence : float, default 0.05
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored.
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands.
        up : array of shape (series, timesteps) 
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'confidence_interval', 'prediction_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'confidence_interval' or interval_type == 'prediction_interval':
            interval_f = eval(interval_type)
            low, up = interval_f(self.data, self.smooth_data, self.X, confidence)

        return low, up



class LowessSmoother(object):

    """
    LowessSmoother uses LOWESS (locally-weighted scatterplot smoothing) 
    to smooth the timeseries. This smoothing technique is a non-parametric 
    regression method that essentially fit a unique linear regression 
    for every data point by including nearby data points to estimate 
    the slope and intercept. The presented method is robust because it 
    performs residual-based reweightings simply specifing the number of 
    iterations to operate.
    
    The LowessSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    smooth_fraction : float
        Between 0 and 1. The smoothing span. A larger value of smooth_fraction 
        will result in a smoother curve.
    iterations : int
        Between 1 and 6. The number of residual-based reweightings to perform.
    copy : bool, default True
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results.
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = LowessSmoother(smooth_fraction=0.3, iterations=1)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, smooth_fraction, iterations=1, copy=True):
        self.smooth_fraction = smooth_fraction
        self.iterations = iterations
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.X = None
        self.__name__ = 'tsmoothie.smoother.LowessSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
               
        Returns
        -------
        self : returns an instance of self
        """

        if self.smooth_fraction >= 1 or self.smooth_fraction <= 0:
            raise ValueError("smooth_fraction must be in the range (0,1)")

        if self.iterations <= 0 or self.iterations > 6:
            raise ValueError("iterations must be in the range (0,6]")

        data = _check_data(data)
        if data.ndim == 1:
            data = data[:,None]
        timesteps = data.shape[0]

        X = np.arange(timesteps)
        w_init = lowess(self.smooth_fraction, timesteps)

        delta = np.ones_like(data)

        for iteration in range(self.iterations):

            w = delta[:,None,:] * w_init[...,None]  # (timesteps, timesteps, n_series)

            wy = w * data[:,None,:]  # (timesteps, timesteps, n_series)
            wyx = wy * X[:,None,None]  # (timesteps, timesteps, n_series)
            wx = w * X[:,None,None]  # (timesteps, timesteps, n_series)
            wxx = wx * X[:,None,None]  # (timesteps, timesteps, n_series)

            b = np.array([wy.sum(axis=0), wyx.sum(axis=0)]).T  # (n_series, timesteps, 2)
            A = np.array([[w.sum(axis=0), wx.sum(axis=0)],
                          [wx.sum(axis=0), wxx.sum(axis=0)]])  # (2, 2, timesteps, n_series)

            XtX = (A.transpose(1,0,2,3)[None,...]*A[:,None,...]).sum(2)  # (2, 2, timesteps, n_series)
            XtX = np.linalg.pinv(XtX.transpose(3,2,0,1))  # (n_series, timesteps, 2, 2)
            XtXXt = (XtX[...,None]*A.transpose(3,2,1,0)[...,None,:]).sum(2)  # (n_series, timesteps, 2, 2)
            betas = np.squeeze(XtXXt @ b[...,None], -1)  # (n_series, timesteps, 2)

            smooth = (betas[...,0] + betas[...,1] * X).T  # (timesteps, n_series)

            residuals = data - smooth
            s = np.median(np.abs(residuals), axis=0).clip(1e-5)  # clip to avoid division by 0
            delta = (residuals / (6.0 * s)).clip(-1, 1)
            delta = np.square(1 - np.square(delta))

        smooth = _check_output(smooth)
        data = _check_output(data)

        self.smooth_data = smooth
        if self.copy:
            self.X = X
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval', 'confidence_interval' and 'prediction_interval'. 
        confidence : float, default 0.05
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored.
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps)
            Lower bands.
        up : array of shape (series, timesteps)
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'confidence_interval', 'prediction_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'confidence_interval' or interval_type == 'prediction_interval':
            interval_f = eval(interval_type)
            low, up = interval_f(self.data, self.smooth_data, self.X, confidence)

        return low, up



class KalmanSmoother(object):

    """
    KalmanSmoother smoothes the timeseries using the Kalman smoothing 
    technique. The Kalman smoother provided here can be represented 
    in the state space form. For this reason, it's necessary to provide 
    an adequate matrix representation of all the components. It's possible 
    to define a Kalman smoother that takes into account the following 
    structure present in our series: 'level', 'trend', 'seasonality' and 
    'long seasonality'. All these features have an addictive behaviour. 
    
    The KalmanSmoother automatically vectorizes, in an efficient way, 
    the desired smoothing operation on all the series received.
    
    Parameters
    ----------
    component : str
        Specify the patterns and the dinamycs present in our series. 
        The possibilities are: 'level', 'level_trend', 
        'level_season', 'level_trend_season', 'level_longseason', 
        'level_trend_longseason', 'level_season_longseason', 
        'level_trend_season_longseason'. Each single component is 
        delimited by the '_' notation. 
    component_noise : dict 
        Specify in a dictionary the noise (in float term) of each single 
        component provided in the 'component' argument. If a noise of a 
        component, not provided in the 'component' argument, is provided, it's 
        automatically ignored. 
    observation_noise : float, default 1.0 
        The noise level generated by the data measurement. 
    n_seasons : int, default None 
        The period of the seasonal component. If a seasonal component is 
        not provided in the 'component' argument, it's automatically ignored. 
    n_longseasons : int, default None 
        The period of the long seasonal component. If a long seasonal component is 
        not provided in the 'component' argument, it's automatically ignored. 
    copy : bool, default True 
        If True, the raw data received by the smoother and the smoothed results 
        can be accessed using 'data' and 'smooth_data' attributes. This is useful 
        to calculate the intervals. If set to False the interval calculation is disabled. 
        In order to save memory, set it to False if you are intereset only 
        in the smoothed results. 
        
    Attributes
    ----------
    smooth_data : array of shape (series, timesteps) 
        Smoothed data derived from the smoothing operation. It is accessible 
        after computhing smoothing, otherwise None is returned.
    data : array of shape (series, timesteps) 
        Raw data received by the smoother. It is accessible with 'copy' = True and 
        after computhing smoothing, otherwise None is returned. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=10, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = KalmanSmoother(component='level_trend', 
                                  component_noise={'level':0.1, 'trend':0.1})
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('kalman_interval')
    """

    def __init__(self, component, component_noise, observation_noise=1.,
                 n_seasons=None, n_longseasons=None, copy=True):
        self.component = component
        self.component_noise = component_noise
        self.observation_noise = observation_noise
        self.n_seasons = n_seasons
        self.n_longseasons = n_longseasons
        self.copy = copy
        self.smooth_data = None
        self.data = None
        self.cov = None
        self.__name__ = 'tsmoothie.smoother.KalmanSmoother'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (series, timesteps) or also (timesteps,) for single timeseries 
            Timeseries to smooth. The data are assumed to be in increasing time order 
            in each timeseries.
        weights : array-like of shape (timesteps,), default None 
            Individual weights for each timestep. In case of multidimesional timeseries, 
            the same weights are used for all the timeseries. 
               
        Returns
        -------
        self : returns an instance of self
        """

        components = ['level', 'level_trend',
                      'level_season', 'level_trend_season',
                      'level_longseason', 'level_trend_longseason',
                      'level_season_longseason', 'level_trend_season_longseason']

        if self.component not in components:
            raise ValueError(f"'{self.component}' is unsupported. Pass one of {components}")

        _noise = _check_noise_dict(self.component_noise, self.component)
        data = _check_data_nan(data)

        if self.component == 'level':

            A = [[1]]  # level
            Q = [[_noise['level']]]
            H = [[1]]

        elif self.component == 'level_trend':

            A = [[1,1],  # level
                 [0,1]]  # trend
            Q = np.diag([_noise['level'], _noise['trend']])
            H = [[1,0]]

        elif self.component == 'level_season':

            if self.n_seasons is None:
                raise ValueError("you should specify n_seasons when using a seasonal component")

            A = np.zeros((self.n_seasons, self.n_seasons))
            A[0,0] = 1 # level
            A[1,1:] = [-1.0] * (self.n_seasons-1)  # season
            A[2:,1:-1] = np.eye(self.n_seasons-2)  # season
            Q = np.diag([_noise['level'],
                         _noise['season']] + [0]*(self.n_seasons-2))
            H = [[1,1] + [0]*(self.n_seasons-2)]

        elif self.component == 'level_trend_season':

            if self.n_seasons is None:
                raise ValueError("you should specify n_seasons when using a seasonal component")

            A = np.zeros((self.n_seasons+1, self.n_seasons+1))
            A[:2,:2] = [[1,1], # level
                        [0,1]] # trend
            A[2,2:] = [-1.0] * (self.n_seasons-1)  # season
            A[3:,2:-1] = np.eye(self.n_seasons-2)  # season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['season']] + [0]*(self.n_seasons-2))
            H = [[1,0,1] + [0]*(self.n_seasons-2)]

        elif self.component == 'level_longseason':

            if self.n_longseasons is None:
                raise ValueError("you should specify n_longseasons when using a long seasonal component")

            period_cycle_sin = np.sin(2*np.pi/self.n_longseasons)
            period_cycle_cos = np.cos(2*np.pi/self.n_longseasons)

            A = [[1,0,0],  # level
                 [0,period_cycle_cos,period_cycle_sin],  # long season
                 [0,-period_cycle_sin,period_cycle_cos]]  # long season
            Q = np.diag([_noise['level'],
                         _noise['longseason'], _noise['longseason']])
            H = [[1,1,0]]

        elif self.component == 'level_trend_longseason':

            if self.n_longseasons is None:
                raise ValueError("you should specify n_longseasons when using a long seasonal component")

            period_cycle_sin = np.sin(2*np.pi/self.n_longseasons)
            period_cycle_cos = np.cos(2*np.pi/self.n_longseasons)

            A = [[1,1,0,0],  # level
                 [0,1,0,0],  # trend
                 [0,0,period_cycle_cos,period_cycle_sin],  # long season
                 [0,0,-period_cycle_sin,period_cycle_cos]]  # long season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['longseason'], _noise['longseason']]),
            H = [[1,0,1,0]]

        elif self.component == 'level_season_longseason':

            if self.n_seasons is None:
                raise ValueError("you should specify n_seasons when using a seasonal component")

            if self.n_longseasons is None:
                raise ValueError("you should specify n_longseasons when using a long seasonal component")

            period_cycle_sin = np.sin(2*np.pi/self.n_longseasons)
            period_cycle_cos = np.cos(2*np.pi/self.n_longseasons)

            A = np.zeros((self.n_seasons+2, self.n_seasons+2))
            A[0,0] = 1  # level
            A[1:3,1:3] = [[period_cycle_cos,period_cycle_sin],  # long season
                          [-period_cycle_sin,period_cycle_cos]]  # long season
            A[3,3:] = [-1.0] * (self.n_seasons-1)  # season
            A[4:,3:-1] = np.eye(self.n_seasons-2)  # season
            Q = np.diag([_noise['level'],
                         _noise['longseason'], _noise['longseason'],
                         _noise['season']] + [0]*(self.n_seasons-2))
            H = [[1,1,0,1] + [0]*(self.n_seasons-2)]

        elif self.component == 'level_trend_season_longseason':

            if self.n_seasons is None:
                raise ValueError("you should specify n_seasons when using a seasonal component")

            if self.n_longseasons is None:
                raise ValueError("you should specify n_longseasons when using a long seasonal component")

            period_cycle_sin = np.sin(2*np.pi/self.n_longseasons)
            period_cycle_cos = np.cos(2*np.pi/self.n_longseasons)

            A = np.zeros((self.n_seasons+2+1, self.n_seasons+2+1))
            A[:2,:2] = [[1,1], # level
                        [0,1]] # trend
            A[2:4,2:4] = [[period_cycle_cos,period_cycle_sin],  # long season
                          [-period_cycle_sin,period_cycle_cos]]  # long season
            A[4,4:] = [-1.0] * (self.n_seasons-1)  # season
            A[5:,4:-1] = np.eye(self.n_seasons-2)  # season
            Q = np.diag([_noise['level'], _noise['trend'],
                         _noise['longseason'], _noise['longseason'],
                         _noise['season']] + [0]*(self.n_seasons-2))
            H = [[1,0,1,0,1] + [0]*(self.n_seasons-2)]


        kf = simdkalman.KalmanFilter(
            state_transition = A,
            process_noise = Q,
            observation_model = H,
            observation_noise = self.observation_noise)

        smoothed = kf.smooth(data)
        smoothed_obs = smoothed.observations.mean
        cov = np.sqrt(smoothed.observations.cov)

        smoothed_obs = _check_output(smoothed_obs, transpose=False)
        cov = _check_output(cov, transpose=False)
        data = _check_output(data, transpose=False)

        self.smooth_data = smoothed_obs
        if self.copy:
            self.cov = cov
            self.data = data

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2, **kwargs):

        """
        Obtain intervals from the smoothed timeseries. 
        Take care to set copy = True when defining the smoother.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported types are 'sigma_interval' and 'kalman_interval'. 
        confidence : float, default 0.05 
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'kalman_interval' is selected 
            as interval_type, otherwise is ignored.
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands.
        up : array of shape (series, timesteps) 
            Upper bands.
        """

        if self.data is None:
            raise ValueError("Pass some data to the smoother before computing intervals, setting copy == True")

        interval_types = ['sigma_interval', 'kalman_interval']

        if interval_type not in interval_types:
            raise ValueError(f"'{interval_type}' is not a supported interval type. Supported types are {interval_types}")

        if interval_type == 'sigma_interval':
            low, up = sigma_interval(self.data, self.smooth_data, n_sigma)
        elif interval_type == 'kalman_interval':
            low, up = kalman_interval(self.data, self.smooth_data, self.cov, confidence)
            return low, up

        return low, up



class WindowWrapper(object):

    """
    WindowWrapper smooths timeseries partitioning them into equal sliding 
    segments and treating them as new standalone timeseries.
    The WindowWrapper handles single timeseries. After the sliding windows are 
    generated, the WindowWrapper smooths them using the smoother it receives as 
    input parameter. In this way, the smoothing can be carried out like a 
    multivariate smoothing task.
    
    The WindowWrapper automatically vectorizes, in an efficient way, 
    the sliding window creation and the desired smoothing operation. 
    
    Parameters
    ----------
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother. 
        It computes the smoothing on the series received.        
    window_shape : int
        Grather than 1. The shape of the sliding windows used to divide 
        the series to smooth.
    step : int, default 1
        The step used to generate the sliding windows.
        
    Attributes
    ----------   
    Smoother : class from tsmoothie.smoother
        Every smoother available in tsmoothie.smoother that was passed to 
        WindowWrapper. It as the same properties and attributes of every Smoother. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from tsmoothie.utils.utils_func import sim_randomwalk
    >>> from tsmoothie.smoother import *

    >>> np.random.seed(33)
    >>> data = sim_randomwalk(n_series=1, timesteps=200, 
                              process_noise=10, measure_noise=30)

    >>> smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.3, iterations=1), 
                                 window_shape=30)
    >>> smoother.smooth(data)

    >>> low, up = smoother.get_intervals('prediction_interval')
    """

    def __init__(self, Smoother, window_shape, step=1):
        self.Smoother = Smoother
        self.window_shape = window_shape
        self.step = step
        self.__name__ = 'tsmoothie.smoother.WindowWrapper'


    def __repr__(self):
        return f"<{self.__name__}>"


    def __str__(self):
        return f"<{self.__name__}>"


    def smooth(self, data):

        """
        Smooth timeseries.
        
        Parameters
        ----------
        data : array-like of shape (1, timesteps) or also (timesteps,) 
            Single timeseries to smooth. The data are assumed to be in increasing time order. 
               
        Returns
        -------
        self : returns an instance of self
        """

        if not hasattr(self.Smoother, '__name__'):
            raise ValueError("Use a Smoother from tsmoothie.smoother")

        if 'tsmoothie.smoother' not in self.Smoother.__name__:
            raise ValueError("Use a Smoother from tsmoothie.smoother")

        data = np.asarray(data)
        if np.prod(data.shape) == np.max(data.shape):
            data = data.ravel()[:,None]
        else:
            raise ValueError("The format of data received is not appropriate. window_wrapper accepts only univariate timeseries")

        if data.shape[0] < self.window_shape:
            raise ValueError("window_shape must be <= than timesteps")

        data = create_windows(data, window_shape=self.window_shape, step=self.step)
        data = np.squeeze(data, -1)

        self.Smoother.smooth(data)

        return self


    def get_intervals(self, interval_type, confidence=0.05, n_sigma=2):

        """
        Obtain intervals from the smoothed timeseries.
        Take care to set copy = True in the Smoother that is passed to WindowWrapper.
        
        Parameters
        ----------
        interval_type : str
            Type of interval used to produced the lower and upper bands. 
            Supported types are the same types supperted by every single Smoother. 
        confidence : float, default 0.05
            The significance level for the intervals calculated as (1-confidence). 
            This parameter is effective only if 'confidence_interval' or 'prediction_interval' 
            are selected as interval_type, otherwise is ignored. 
        n_sigma : int, default 2
            How many standard deviations, calculated on residuals of the smoothing operation, 
            are used to obtain the intervals. This parameter is effective only if 'sigma_interval' 
            is selected as interval_type, otherwise is ignored.
               
        Returns
        -------
        low : array of shape (series, timesteps) 
            Lower bands. 
        up : array of shape (series, timesteps) 
            Upper bands. 
        """

        if self.Smoother.data is None:
            raise ValueError("Pass some data to the WindowWrapper smoother before computing intervals, setting copy == True in the Smoother")

        low, up = self.Smoother.get_intervals(interval_type, confidence=confidence, n_sigma=n_sigma)

        return low, up