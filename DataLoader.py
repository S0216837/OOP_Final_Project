import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self):
        self._data = None
        self._X = None
        self._Y = None

    @abstractmethod
    def loader(self, path_filename):
        raise NotImplementedError

    def print_data(self):
        print('Data, the first 5 lines: \n', self._data.head())
        print('\n')
        print('Data info: ')
        self._data.info()
        print('\n')
        print('Data description: \n', self._data.describe())

    def set_X(self, column_names):
        if self._data is None:
            raise ValueError("Data is not available, run loader first.")
        missing_columns = [col for col in column_names if col not in self._data.columns]
        if missing_columns:
            raise ValueError(f"These columns do not exist in the dataframe: {missing_columns}")
        self._X = self._data[column_names]
        print(f'X is set to be: \n{self._X}')

    def get_X(self):
        return self._X

    def set_Y(self, column_name):
        if column_name in self._data.columns:
            self._Y = self._data[column_name]
            print(f'Y is set to be: {column_name}')
        else:
            raise ValueError(f"Variable {column_name} does not exist in the dataframe.")

    def get_Y(self):
        return self._Y

    def add_constant(self):
        if self._X is not None and 'const' not in self._X.columns:
            self._X.insert(0, 'const', 1.0)
        print(f'X after adding constant: \n{self._X}')

    def x_transpose(self):
        if self._X is None:
            raise ValueError("X is not set, please set X first")
        return self._X.T 

class CsvLoader(DataLoader):
    def loader(self, path_filename):
        data = pd.read_csv(path_filename)
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')
        self._data = data
        print('Data has been loaded successfully')
        return self._data

class StatsmodelsLoader(DataLoader):
    def loader(self, dataset):
        data = dataset.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')
        self._data = data
        print('Data has been loaded successfully')
        return self._data


