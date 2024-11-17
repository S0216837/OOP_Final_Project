import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize

##
#  
class GLM:
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

    # @abstractmethod implement this abstrctmethod so that in subclass it can be overriden and implemented in details
    def predict(self, X_new):
        raise NotImplementedError

    def get_llik(self):
        if self._params is None:
            raise ValueError ('Model parameters are not yet estimated.')
        else:
            llik = self.llik_func(self._params)
            return llik
            

class NormalModel(GLM):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._model = 'Normal'

    def llik_func(self, params):
        eta = np.matmul(self._X, params)
        mu = eta   # Identity link function
        llik = np.sum(norm.logpdf(self._Y, mu))
        self._llik = llik
        return llik

    def predict(self, X_new):
        if self._params is None:
            raise ValueError ('Model parameters are not yet estimated.')
        X_new = self._X.tail(3)
        # mu = eta  Identity link function
        predictions = np.matmul(X_new, self._params) 
        return predictions

class BernoulliModel(GLM):
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

    def predict(self, X_new):
        if self._params is None:
            raise ValueError ('Model parameters are not yet estimated.')
        X_new = self._X.tail(3)
        eta = np.matmul(X_new, self._params)
        predictions = 1 / (1 + np.exp(-eta))
        return predictions
        

class PoissonModel(GLM):
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

    def predict(self, X_new): 
        if self._params is None: 
            raise ValueError('Model parameters are not yet estimated.')
        X_new = self._X.tail(3)
        eta = np.matmul(X_new, self._params) 
        predictions = np.exp(eta) # Exponential function return predictions
        return predictions


def main(model_type):
    
   
    ##
    # load the dataset based on teh model type parameter
    # define Y,the dependent variable, 
    # define X, independent variable matrix and add constant as an intercept
    ##
    #Create an instance of the model class based on the passed parameter
    #
    try: 
        if model_type == 'Normal':
            data = sm.datasets.get_rdataset("Duncan", "carData")
            Y = data.data['income']
            X = data.data[['education', 'prestige']]
            X = sm.add_constant(X)
            model_t = NormalModel(X, Y)
            
        elif model_type == 'Bernoulli':
            data = sm.datasets.spector.load_pandas()
            Y = data.data['GRADE']
            X = data.data[['GPA', 'TUCE','PSI']]
            X = sm.add_constant(X)
            model_t = BernoulliModel(X, Y)
        
        elif model_type == "Poisson": 
            data = pd.read_csv('https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv')
            Y = data['breaks']
            X = data[['wool', 'tension']]
            X = sm.add_constant(X)
            model_t = PoissonModel(X, Y)
            
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose from'Normal', 'Bernoulli', or 'Poisson'.") 
    except ValueError as e:
         print(e)
         return 

    # Fit the model using X, Y to estimate parameters and predict using X_new
    try: 
        Model_name = model_t.getModel() 
        print('The model name is:', Model_name, end="\n\n")
        print('Y dependent variable is: ', Y.name, end = "\n\n")
        print('X independent variables are: \n', X.columns.tolist(), end = "\n\n")

        # Fit the model to estimate parameters
        estimated_parameters = model_t.fit()

        # Print the estimated parameters 
        print(f'Estimated parameters are: {estimated_parameters}', end="\n\n") 

        # Calculate the max log-likelihood of the data and print the result 
        llik_value = model_t.get_llik() 
        print(f'Maximum log-likelihood of the data is: {llik_value}', end="\n\n")

        # assign X_new using the last 3 rows from the X matrix, use the predict method on X_new.
        X_new = X.tail(3)
        predictions = model_t.predict(X_new)
        print(f'Predictions for the new data are: \n{predictions}', end="\n\n")
        
    except Exception as e: 
        print(f"An error occured during model fitting or prediction: {e}")

