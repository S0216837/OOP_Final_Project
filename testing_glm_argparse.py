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


def main(model_type, dset, predictors, add_intercept):
    model_types = {'normal': NormalModel, 'bernoulli': BernoulliModel, 'poisson':PoissonModel}
    statsmodels_families = {'normal': sm.families.Gaussian(), 'bernoulli': sm.families.Binomial(), 'poisson': sm.families.Poisson()}

    if model_type not in model_types: 
        raise ValueError(f"Invalid model type: {model_type}. Please choose from 'normal', 'bernoulli' or 'poisson'.")
    
    data, X, Y = load_dataset(dset, predictors, add_intercept)

    model_instance = model_types[model_type](X, Y)
    statsmodel_instance = sm.GLM(Y, X, family = statsmodels_families[model_type]).fit()

    print(f"\nTesting {model_type.capitalize()} Model\n")

    # Fit the model
    estimated_params = model_instance.fit()
    print(f"Estimated parameters: {estimated_params}")

    # Predict using the fitted model
    X_new = X.tail(3)
    predictions = model_instance.predict(X_new)
    print(f"Predictioons for new data: \n{predictions}")

    # Get log-likelihood
    llik = model_instance.get_llik()
    print(f"Log-likelihood: {llik}")

    # Compare with statsmodels results
    print("\nComparing with statsmodels results:")
    sm_params = statsmodel_instance.params
    print(f"Statsmodels estimated parameters: {sm_params.values}")

    sm_pred = statsmodel_instance.predict(X)
    print(f"Statsmodels predictions: \n{sm_pred.tail(3).values}")


# load data from commandline inputs
def load_dataset(dset, predictors, add_intercept):
    if dset == 'duncan':
        data = sm.datasets.get_rdataset("Duncan", "carData").data
        Y = data['income']
        X = data[predictors]
    elif dset == 'spector':
        data = sm.datasets.spector.load_pandas().data
        Y = data['GRADE']
        X = data[predictors]
    elif dset == 'warpbreaks':
        data = pd.read_csv('https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv')
        Y = data['breaks']
        X = data[predictors]
    else:
        raise ValueError(f"Invalid dataset: {dset}. Please choose from 'duncan', 'spector', or 'warpbreaks'.")

    if add_intercept:
        X = sm.add_constant(X)

    return data, X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized Linear Model")
    parser.add_argument('--model', type=str, required=True, choices=['normal', 'bernoulli', 'poisson'], help='Model type to use')
    parser.add_argument('--dset', type=str, required=True, choices=['duncan', 'spector', 'warpbreaks'], help='Dataset to use')
    parser.add_argument('--predictors', type=str, nargs='+', required=True, help='List of predictor variables')
    parser.add_argument('--add_intercept', action='store_true', help='Add intercept to the model')

    args = parser.parse_args()
    main(args.model, args.dset, args.predictors, args.add_intercept)

 