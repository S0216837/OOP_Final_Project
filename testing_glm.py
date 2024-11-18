##
# testing file for GLM_main and the results are compared with the in-build functions in statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize
import argparse
# import my classes here

from GLMs import GLM, NormalModel, BernoulliModel, PoissonModel


def main():
    models= {'Normal': NormalModel, 'Bernoulli': BernoulliModel, 'Poisson': PoissonModel}
    model_types = ['Normal', 'Bernoulli', 'Poisson']

    for model_type in model_types:
        print(f"\n Testing {model_type}", end="\n\n")
        model_instance, statsmodel_instance = get_model_instance(model_type)
        
        # Fit the model
        estimated_params = model_instance.fit()
        print(f"Estimated parameters for {model_type} model: ", estimated_params)

        # Predict using the fitted model
        X_new = model_instance._X.tail(3)
        predictions = model_instance.predict(X_new)
        print(f"Predictions for new data in {model_type} model:\n", predictions)

        # Get log-likelihood
        llik = model_instance.get_llik()
        print(f"Log-likelihood for {model_type} model: ", llik)

        # Compare with statsmodels results
        print("\nComparing with statsmodels results: ")
        sm_params = statsmodel_instance.params
        print(f"Statsmodels estimated parameters for {model_type} model: ", sm_params.values)

        sm_pred = statsmodel_instance.predict(model_instance._X)
        print(f"Statsmodels predictions for {model_type} model:\n ", sm_pred.tail(3).values)


def get_model_instance(model_type): 
    #Define and load dataset, define X and Y
    if model_type == 'Normal':
        data = sm.datasets.get_rdataset("Duncan", "carData")
        Y = data.data['income']
        X = data.data[['education', 'prestige']]
        X = sm.add_constant(X)
        model_instance = NormalModel(X, Y)
        statsmodel_instance = sm.GLM(Y, X, family = sm.families.Gaussian()).fit()
    elif model_type == 'Bernoulli':
        data = sm.datasets.spector.load_pandas()
        Y = data.data['GRADE']
        X = data.data[['GPA', 'TUCE','PSI']]
        X = sm.add_constant(X)
        model_instance = BernoulliModel(X, Y)
        statsmodel_instance = sm.GLM(Y, X, family = sm.families.Binomial()).fit()
    elif model_type == 'Poisson':
        data = pd.read_csv('https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv')
        Y = data['breaks']
        X = data[['wool', 'tension']]
        X = sm.add_constant(X)
        model_instance = PoissonModel(X, Y)
        statsmodel_instance = sm.GLM(Y, X, family = sm.families.Poisson()).fit()

    else: 
         raise ValueError(f"Invalid model type: {model_type}. Please choose from 'Normal', 'Bernoulli' or 'Poisson'")
    
    return model_instance, statsmodel_instance

if __name__ == "__main__":
     main()
    
