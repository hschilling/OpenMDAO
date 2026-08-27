from time import perf_counter
import numpy as np
import pandas as pd

import openmdao.api as om
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.surrogate_models.nearest_neighbor import NearestNeighbor
from openmdao.surrogate_models.response_surface import ResponseSurface
from openmdao.surrogate_models.tests.test_kriging import branin


# ==============================================================================
# read in the data from Kris Pierson
# ==============================================================================
def read_function_data(filepath):
    """
    Read a CSV file containing function inputs and outputs.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    output_names : list
        Names of the output columns (columns 2-3)
    output_data : numpy.ndarray
        Output values as float with shape (n, 2) where n is number of data points
    input_names : list
        Names of the input columns (columns 4-7)
    input_data : numpy.ndarray
        Input values as float with shape (n, 4) where n is number of data points
    """
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Check for "FailedSim" in the second column (index 1)
    failed_mask = df.iloc[:, 1] == "FailedSim"
    num_failed = failed_mask.sum()

    if num_failed > 0:
        print(f"WARNING: Found {num_failed} row(s) with 'FailedSim' in column '{df.columns[1]}' in file '{filepath}'. These rows will be dropped.")
        # Drop rows with FailedSim
        df = df[~failed_mask].reset_index(drop=True)

    # Get column names (excluding the first case number column)
    all_columns = df.columns.tolist()

    # Extract output columns (indices 1 and 2, which are columns 2-3 in the file)
    output_names = all_columns[1:3]
    output_data = df.iloc[:, 1:3].values.astype(float)  # Convert to float, shape (n, 2)

    # num_inputs = df.shape[1] - 3 # minus case and output columns

    # Extract input columns
    input_names = all_columns[3:]
    input_data = df.iloc[:, 3:].values.astype(float)  # Convert to float, shape (n, 4)

    return output_names, output_data, input_names, input_data


output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-4DVs/Results_500-4DVs-Train.csv')
output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-4DVs/Results_2K-4DVs-GlobalTest.csv')

# output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-6DVs/Results_500-Train.csv')
# output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-6DVs/Results_2K-GlobalTest.csv')


# output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-8DVs/Results_500-8DVs-Train.csv')
# output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-8DVs/Results_2K-8DVs-GlobalTest.csv')



# x = np.array([[-2., 0.], [-0.5, 1.5], [1., 1.], [0., .25], [.25, 0.], [.66, .33]])
# y = np.array([[branin(case)] for case in x])

# surrogate = ResponseSurface()
# surrogate.train(x, y)

# x0 = x[0]
# mu = surrogate.predict(x0)

# mu = surrogate.predict(np.array([.5, .5]))



# surrogate = KrigingSurrogate()
surrogate = ResponseSurface()
# surrogate = NearestNeighbor()
surrogate.train(input_train, output_train)


# Get a single input sample (first row of your test data)
single_input = input_test[0:1]  # Keep it as (1, num_features) shape


single_input = single_input.squeeze()

# Warmup prediction (first prediction is often slower due to graph compilation)
_ = surrogate.predict(single_input)

# Time a single prediction
start_time = perf_counter()
output_predicted = surrogate.predict(single_input)
end_time = perf_counter()

inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Single inference time: {inference_time:.2f} ms")

# For more accurate timing, run multiple predictions and average
num_runs = 100
times = []

for i in range(num_runs):
    start_time = perf_counter()
    _ = surrogate.predict(single_input)
    end_time = perf_counter()
    times.append((end_time - start_time) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)
print(f"\nAverage inference time over {num_runs} runs: {avg_time:.3f} ± {std_time:.3f} ms")
print(f"Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")


