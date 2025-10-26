import re
import numpy as np

def extract_values_from_string(input_string):
    # Define the pattern to match the values
    pattern = r'dEW-(\d+(\.\d+)?)_snCut-(\d+(\.\d+)?)_attempt-(\d+(\.\d+)?)'

    # Use regular expression to find matches
    match = re.search(pattern, input_string)

    # If match found, extract values
    if match:
        value1 = float(match.group(1))
        value2 = float(match.group(3))
        value3 = float(match.group(5))
        return value1, value2, value3
    else:
        return None
    
def calculate_chi_square(x_data, y_data, y_err):
    # Define the 1:1 line model (y = x)
    expected_y = x_data
    
    # Calculate the residuals
    residuals = (y_data - expected_y) / y_err
    
    # Calculate the chi-square value
    chi_square = np.sum(residuals ** 2)
    
    return chi_square

def calculate_reduced_chi_square(x_data, y_data, y_err, num_params=2):
    # Calculate the chi-square value
    chi_square = calculate_chi_square(x_data, y_data, y_err)
    
    # Calculate the degrees of freedom
    dof = len(x_data) - num_params
    
    # Calculate the reduced chi-square value
    reduced_chi_square = chi_square / dof
    
    return reduced_chi_square