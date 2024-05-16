import os
import numpy as np
import pandas as pd
import re

def parse_dir_name(dir_name):
    """
    Parse the directory name to extract parameters and their values.
    
    Args:
    - dir_name (str): The directory name to parse.
    
    Returns:
    - dict: A dictionary where keys are parameter names and values are parameter values.
    """
    # Split the directory name by '_' and parse
    parts = dir_name.split('_')
    params = {}
    for i in range(0, len(parts), 2):
        key = parts[i]
        try:
            value = parts[i + 1]
            # Attempt to convert numeric values
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            value = value  # Keep as string if conversion fails
        except IndexError:
            continue  # Skip if there's no value for a key
        params[key] = value
    return params

def load_dataframes(root_dir):
    """
    Load numpy arrays from subdirectories and return a pandas DataFrame.
    
    Args:
    - root_dir (str): The root directory to search recursively.
    
    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a subdirectory's attributes and loaded numpy array.
    """
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                # Load the numpy array
                npy_path = os.path.join(subdir, file)
                npy_array = np.load(npy_path)
                
                # Parse the directory name to get parameters
                dir_name = os.path.relpath(subdir, root_dir)
                params = parse_dir_name(dir_name.replace('/', '_'))
                
                # Add the numpy array to params dict
                params['numpyArray'] = npy_array
                
                # Append to data list
                data.append(params)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage
if __name__ == "__main__":
    root_dir = "experiments/"
    df = load_dataframes(root_dir)
    print(df)
