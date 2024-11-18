import pandas as pd
import statsmodels.api as sm
from DataLoader import DataLoader, CsvLoader, StatsmodelsLoader


def test_file_loader():
    file_loader = CsvLoader()
    data = file_loader.loader('https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv')
    file_loader.print_data()
    file_loader.set_X(['wool', 'tension'])
    file_loader.set_Y('breaks')
    file_loader.add_constant()
    file_loader.print_data()
    print("FileLoader test completed successfully.\n")

def test_statsmodels_loader_with_wrong_columns():
    try:
        test_wrong_col = StatsmodelsLoader()
        test_wrong_col.loader(sm.datasets.get_rdataset('Duncan', 'carData'))
        test_wrong_col.print_data()
        test_wrong_col.set_X(['wool', 'tension'])
        test_wrong_col.set_Y('breaks')
    except ValueError as e:
        print(f"StatsmodelsLoader test with wrong columns failed as expected: {e}\n")

def test_statsmodels_loader_with_correct_columns():
    test_ok = StatsmodelsLoader()
    test_ok.loader(sm.datasets.get_rdataset('Duncan', 'carData'))
    test_ok.print_data()
    test_ok.set_X(['education', 'prestige'])
    test_ok.set_Y('income')
    test_ok.add_constant()
    test_ok.print_data()
    x_data = test_ok.get_X()
    y_data = test_ok.get_Y()
    print("StatsmodelsLoader test with correct columns completed successfully.\n")
    print(f"X data: \n{x_data}\n")
    print(f"Y data: \n{y_data}\n")

if __name__ == "__main__":
    print("Running tests...\n")
    test_file_loader()
    test_statsmodels_loader_with_wrong_columns()
    test_statsmodels_loader_with_correct_columns()
    print("All tests completed.")
