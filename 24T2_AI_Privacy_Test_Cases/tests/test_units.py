import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CarPurchaseUnits import load_data, preprocess_data, split_data
import pandas as pd
import numpy as np

#Test case to ensure the correct loading of data
def test_load_data():
    data = load_data('Car_Purchasing_Data.xlsx')
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame."

    expected_columns = ['Customer Name', 'Customer e-mail', 'Country', 'Gender', 
                        'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']

    for col in expected_columns:
        assert col in data.columns, f"Expected column {col} not found in loaded data."

#Test case to ensure the correct shape of data
def test_shape_of_data():
    data = load_data('Car_Purchasing_Data.xlsx')
    X_scaled, y_scaled, _, _ = preprocess_data(data)
    
    #expected number of columns
    assert X_scaled.shape[1] == 5, "Expected 5 features in the X data after preprocessing."
    assert y_scaled.shape[1] == 1, "Expected Y data to have a single column."
    #expected number of rows
    assert X_scaled.shape[0] == 500, "Expected 5 features in the X data after preprocessing."
    assert y_scaled.shape[0] == 500

#Test case to ensure the correct columns for Input
def test_columns_X():
    data = load_data('Car_Purchasing_Data.xlsx')
    X, _, _, _ = preprocess_data(data)
    input_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    # Convert the NumPy array to a DataFrame
    X_df = pd.DataFrame(X, columns=input_columns)
    # Check if X_df is a DataFrame
    assert isinstance(X_df, pd.DataFrame)
    # Check that the columns have been dropped for X
    assert "Customer Name" not in X_df.columns
    assert "Customer e-mail" not in X_df.columns
    assert "Country" not in X_df.columns
    assert "Car Purchase Amount" not in X_df.columns

#Test case to ensure the correct column for output
def test_columns_Y():
    data = load_data('Car_Purchasing_Data.xlsx')
    _, Y, _, _ = preprocess_data(data)
    # Convert the NumPy array to a DataFrame
    Y_df = pd.DataFrame(Y, columns=['Car Purchase Amount'])
    
    # Check if Y_df is a DataFrame and has the correct column name
    assert isinstance(Y_df, pd.DataFrame)
    assert Y_df.columns == 'Car Purchase Amount'
    assert 'Car Purchase Amount' in Y_df.columns