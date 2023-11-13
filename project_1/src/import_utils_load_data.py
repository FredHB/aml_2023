# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# sklearn machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

# Read data using pandas
read_path = './data/processed/'
X = pd.read_csv(read_path + 'X_train.csv')
X_test = pd.read_csv(read_path + 'X_test.csv')
y = pd.read_csv(read_path + 'Y_train.csv')
ids = X_test['id'].copy() # Save test ids for submission
X.drop('id', axis=1, inplace=True)
X_test.drop('id', axis=1, inplace=True)
y.drop('id', axis=1, inplace=True)

# utility functions
def validate_model(y_val, y_pred):
    """
    Calculates the mean squared error and R^2 score between the predicted and actual values of the validation set.
    Then, plots the predicted vs actual values.

    Parameters:
    y_val (array-like): The actual values of the validation set.
    y_pred (array-like): The predicted values of the validation set.

    Returns:
    None
    """    
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")
    print(f"R^2 on Validation Set: {r2}")

    # Plotting predicted vs actual values
    plt.figure(figsize=(6, 3))
    plt.scatter(y_val, y_pred, alpha=0.7)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)  # Diagonal line
    plt.show()

def model_print_results(X_train, model, filename) :
    """
    Summarizes the results of the LASSO regression model by printing the non-zero coefficients and their corresponding variable names.
    Then, writes the results to a text file in the 'out/model' directory.

    Parameters:
    X_train (DataFrame): The training set features.
    model (Lasso): The trained LASSO regression model.
    filename (str): The name of the output file.

    Returns:
    None
    """
    colnames = np.array(X_train.columns.to_list())

    # Extracting non-zero coefficients
    non_zero_coefs = model.coef_[model.coef_ != 0]
    non_zero_varnames = colnames[model.coef_ != 0]
    non_zero_coefs = pd.DataFrame({'Variable': non_zero_varnames, 'Coefficient': non_zero_coefs})

    # Set pandas options to print full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Summarizing the results
    # Create the output directory if it doesn't exist
    os.makedirs("out/model", exist_ok=True)

    # Define the output file path
    output_file = f"out/model/{filename}.txt"

    # Write the lines to the output file
    with open(output_file, "w") as f:
        f.write("Non-zero LASSO Coefficients:\n")
        f.write(str(non_zero_coefs) + "\n")
        f.write(f"Total number of non-zero coefficients: {len(non_zero_coefs)}\n")

    # Reset pandas options to default
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

def lasso_cv_error_path(model):
    """
    Plots the mean squared error (MSE) of the LASSO regression model as a function of the regularization parameter alpha.
    The plot shows the cross-validation error path, which is the average MSE across all folds for each alpha value.

    Parameters:
    model (LassoCV): The trained LASSO regression model.

    Returns:
    None
    """
    
    # Cross-validation scores
    mean_mse = np.mean(model.mse_path_, axis=1)
    alphas = model.alphas_

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.plot(alphas, mean_mse, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Parameter)')
    plt.ylabel('Mean Squared Error')
    plt.title('LASSO Cross-Validation Error Path')
    plt.gca().invert_xaxis()  # Invert x-axis to show smaller alpha values on the right
    plt.show()