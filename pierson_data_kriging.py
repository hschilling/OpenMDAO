import numpy as np
import pandas as pd

import openmdao.api as om


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

prob = om.Problem()

# FS	FS	Mass	lean_050	lean_100	sweep_050	sweep_100

pierson_mm = om.MetaModelUnStructuredComp()
for name in input_names:
    pierson_mm.add_input(name, 0.)
for name in output_names:
    pierson_mm.add_output(name, 0., surrogate=om.KrigingSurrogate())

prob.model.add_subsystem('pierson_mm', pierson_mm)

prob.setup()

# train the surrogate and check predicted value
for i, name in enumerate(input_names):
    pierson_mm.options[f'train_{name}'] = input_train[:,i]
for i, name in enumerate(output_names):
    pierson_mm.options[f'train_{name}'] = output_train[:,i]

# test using training data. try just one point at first
for i, name in enumerate(input_names):
    prob.set_val(f'pierson_mm.{name}', input_test[0, i])

from time import perf_counter

start_time = perf_counter()
prob.run_model()
end_time = perf_counter()

print(f"fit time = {end_time-start_time}")

# compare actual to predicted
output_predicted = np.empty(output_test.shape)

for i_case in range(output_test.shape[0]):
    # set inputs
    for i, name in enumerate(input_names):
        prob.set_val(f'pierson_mm.{name}', input_test[i_case, i])
    prob.run_model()
    outputs = []
    for name in output_names:
        outputs.append(prob.get_val(f'pierson_mm.{name}')[0])
    output_predicted[i_case] = outputs

# Mean Absolute Error (MAE) for each output
mae_per_output = np.mean(np.abs(output_test - output_predicted), axis=0)

# Display results
print("Mean Absolute Error per output:")
for i, name in enumerate(output_names):
    print(f"{name}: {mae_per_output[i]:.6f}")

# for i, name in enumerate(output_names):
#     print(f"actual {name} = {output_test[0, i]}")
#     predicted_output = prob.get_val(f'pierson_mm.{name}')[0]
#     print(f"predicted {name} = {predicted_output}")

# try all the test data values. make an array row by
