import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize

##
#  
class GLMs:
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y
        self._params = None
        self._llik = None
        self._model = None

    ##
    # @abstractmethod implement this abstrctmethod so that in subclass it can be overriden and implemented in details
    #
    def llik_func(self, params):
        raise NotImplementedError

    def negllik_func(self, params):
        return -self.llik_func(params)

    def fit(self):
        num_params = self._X.shape[1]
        init_params = np.repeat(0.1, num_params)
        results = minimize(self.negllik_func, init_params)

        print('Optimization is finished. Estimated parameters:', results['x'])
        print(results)
        self._params = results['x']
        return results['x']


    def getParams(self):
        return self._params

    def getModel(self):
        return self._model

    def predict(self, X_new):
        if self._params is None:
            raise ValueError ('Model parameters are not yet estimated.')
        X_new = sm.add_constant(X_new)
        predictions = np.matmul(X_new, self._params)
        return predictions

    def get_llik(self):
        if self._params is None:
            raise ValueError ('Model parameters are not yet estimated.')
        else:
            llik = self.llik_func(self._params)
            return llik
            

class NormalModel(GLMs):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._model = 'Normal'

    def llik_func(self, params):
        eta = np.matmul(self._X, params)
        mu = eta   # Identity link function
        llik = np.sum(norm.logpdf(self._Y, mu))
        self._llik = llik
        return llik

class BernoulliModel(GLMs):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._model = 'Bernoulli'

    def llik_func(self, params):
        eta = np.matmul(self._X, params)
        # use the logistic function as the link function for Bernoulli distribution    
        mu = 1 / (1 + np.exp(-eta))
        llik = np.sum(bernoulli.logpmf(self._Y, mu))
        self._llik = llik
        return llik

class PoissonModel(GLMs):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._model = 'Poisson'

    def llik_func(self, params):
        eta = np.matmul(self._X, params)
        # use the exponential function as the link function for Poisson distribution
        mu = np.exp(eta)
        llik = np.sum(poisson.logpmf(self._Y, mu))
        self._llik = llik
        return llik