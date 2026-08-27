import random
from time import perf_counter
import os

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

# Define a chain of sklearn preprocessing transformers for input data
sklearn_input_transformers = [
    # Step 1: Standardize features (mean=0, std=1) - Z-score normalization
    StandardScaler(copy=True, with_mean=True, with_std=True),
    # Step 2: Scale to range [-2, +2] after standardization
    MinMaxScaler(feature_range=(-2.0, +2.0), copy=True, clip=False),
]

# Define preprocessing transformers for output data (targets)
sklearn_output_transformers = [
    # Step 1: Standardize outputs (mean=0, std=1)
    StandardScaler(copy=True, with_mean=True, with_std=True),
    # Step 2: Scale to range [-1, +1] - smaller range than inputs
    MinMaxScaler(feature_range=(-1.0, +1.0), copy=True, clip=False),
]

def set_tf_seed():
    """
    Set several seeds used by TensorFlow, to ensure the reproducibility of the results.
    """
    os.environ["PYTHONHASHSEED"] = str(0)
    random.seed(1)
    np.random.seed(2)
    tf.random.set_seed(3)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # It has probably already been set.
        pass

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

    # Extract input columns
    input_names = all_columns[3:]
    input_data = df.iloc[:, 3:].values.astype(float)  # Convert to float, shape (n, 4)

    return output_names, output_data, input_names, input_data

def sklearn_fit_transform(data, sklearn_transformers):
    data_transformed = data.copy() # Start with original data
    for transformer in sklearn_transformers:
        # Each transformer learns parameters from data and transforms it
        data_transformed = transformer.fit_transform(data_transformed)
    return data_transformed

def sklearn_transform(data, sklearn_transformers):
    data_transformed = data.copy() # Start with original data
    for transformer in sklearn_transformers:
        # Each transformer learns parameters from data and transforms it
        data_transformed = transformer.transform(data_transformed)
    return data_transformed

def sklearn_reverse_transform(data, sklearn_transformers):
    data_transformed = data.copy() # Start with original data
    for transformer in reversed(sklearn_transformers):
        # Each transformer learns parameters from data and transforms it
        data_transformed = transformer.inverse_transform(data_transformed)
    return data_transformed

# ==============================================================================
# READ DATA
# ==============================================================================
# output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-4DVs/Results_500-4DVs-Train.csv')
# output_names, output_valid, input_names, input_valid = read_function_data( 'kris_pierson_data/Data-4DVs/Results_500-4DVs-Valid.csv')
# output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-4DVs/Results_2K-4DVs-GlobalTest.csv')

# output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-6DVs/Results_500-Train.csv')
# output_names, output_valid, input_names, input_valid = read_function_data( 'kris_pierson_data/Data-6DVs/Results_500-Valid.csv')
# output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-6DVs/Results_2K-GlobalTest.csv')

output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-8DVs/Results_500-8DVs-Train.csv')
output_names, output_valid, input_names, input_valid = read_function_data( 'kris_pierson_data/Data-8DVs/Results_500-8DVs-Valid.csv')
output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-8DVs/Results_2K-8DVs-GlobalTest.csv')

# output_names, output_train, input_names, input_train = read_function_data( 'kris_pierson_data/Data-8DVs/Results_5K-8DVs-Train.csv')
# output_names, output_valid, input_names, input_valid = read_function_data( 'kris_pierson_data/Data-8DVs/Results_5K-8DVs-Valid.csv')
# output_names, output_test, input_names, input_test = read_function_data( 'kris_pierson_data/Data-8DVs/Results_2K-8DVs-GlobalTest.csv')


n_inputs = len(input_names)
n_outputs = len(output_names)

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
# use the training data to learn mean/std/min/max. So do a fit_transform here, not just transform
input_train_transformed = sklearn_fit_transform(input_train, sklearn_input_transformers)
output_train_transformed = sklearn_fit_transform(output_train, sklearn_output_transformers)

# now that the transforms have learned the mean/std/min/max from the training data, apply the transforms to the validation data
input_valid_transformed = sklearn_transform(input_valid, sklearn_input_transformers)
output_valid_transformed = sklearn_transform(output_valid, sklearn_output_transformers)

# ==============================================================================
# NEURAL NETWORK MODEL DEFINITION
# ==============================================================================
# Set TensorFlow random seed for reproducibility (custom function)
set_tf_seed()
# Create a Sequential neural network model
if True:
    tf_model = tf.keras.Sequential(
        [
            # Input layer: explicitly define input shape (2 features)
            tf.keras.Input(shape=(n_inputs,)),
            # Hidden layer 1: 5 neurons with tanh activation
            # tanh outputs values in range [-1, 1], good for normalized data
            tf.keras.layers.Dense(2**7, activation="leaky_relu",
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                    ),
            tf.keras.layers.Dropout(0.1),  # Add dropout after first hidden layer
            # Hidden layer 2: 5 neurons with tanh activation
            # Small network since we only need "smooth result" as comment mentions
            tf.keras.layers.Dense(2**7, activation="leaky_relu",
                                    # kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                    ),
            tf.keras.layers.Dropout(0.1),  # Add dropout after second hidden layer
            # Output layer: 3 neurons (one for each output), no activation (linear)
            # Linear activation is typical for regression problems
            tf.keras.layers.Dense(n_outputs),
        ]
    )





# tf_model = tf.keras.Sequential([
#     tf.keras.Input(shape=(n_inputs,)),
#     tf.keras.layers.Dense(32, activation="leaky_relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(n_outputs),
# ])



# Configure the model for training
tf_model.compile(
    # Adam optimizer with learning rate 0.01 - adaptive learning rate algorithm
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    # Mean Squared Error loss - standard for regression problems
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_mean_absolute_error',
    baseline=0.14,  # Average MAE across all outputs
    patience=50,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# ==============================================================================
# MODEL TRAINING
# ==============================================================================
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=1e-6)

start_time = perf_counter()
hist = tf_model.fit(
    x=input_train_transformed,
    y=output_train_transformed,
    epochs=500,
    batch_size=32,
    verbose=0,
    validation_data=(input_valid_transformed,
                     output_valid_transformed),
    callbacks=[reduce_lr, early_stop],
)
end_time = perf_counter()

print(f"fit time = {end_time-start_time}")


# ==============================================================================
# MODEL TRAINING HISTORY
# ==============================================================================
history = hist.history

print(f"Final training loss: {history['loss'][-1]:.6f}")
print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

# Check what metrics are now available
print("Available metrics:", hist.history.keys())


# Plot training history
import matplotlib.pyplot as plt
# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on primary (left) y-axis
ax1.plot(history['loss'], label='Training Loss', color='tab:blue')
ax1.plot(history['val_loss'], label='Validation Loss', color='tab:orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='black')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Create secondary (right) y-axis
ax2 = ax1.twinx()
ax2.plot(history['val_mean_absolute_error'], label='Validation MAE',
         color='tab:green', linestyle='--', linewidth=2)
ax2.set_ylabel('Validation MAE', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.legend(loc='upper right')

# Optional: set log scale for MAE too if needed
# ax2.set_yscale('log')

plt.title('Training Progress')
plt.tight_layout()
plt.show()

# # ==============================================================================
# # # TRANSFORM TESTING DATA
# # ==============================================================================
input_test_transformed = sklearn_transform(input_test, sklearn_input_transformers)
# output_test_transformed = sklearn_transform(output_test, sklearn_output_transformers)

# test using the test data set
output_predicted_transformed = tf_model.predict(
    input_test_transformed,            # Use preprocessed input data
    batch_size=len(input_test_transformed),    # Process all samples at once
    verbose=0                     # Don't print prediction progress
)

# Convert predictions back to original output scale
# Start with the transformed predictions
output_predicted = sklearn_reverse_transform(output_predicted_transformed, sklearn_output_transformers)

absolute_difference = np.absolute(output_test - output_predicted)
# keep track of the average error
# error = np.sum(absolute_difference)

if True:
    print("Row | Actual                          | Predicted                       | Error")
    print("-" * 85)
    # for i in range(output_test.shape[0]):
    for i in range(5):
        actual = output_test[i]
        predicted = output_predicted[i]
        error = np.abs(actual - predicted)
        print(f"{i:3d} | [{actual[0]:8.4f}, {actual[1]:8.4f}] | "
            f"[{predicted[0]:8.4f}, {predicted[1]:8.4f}] | "
            f"[{error[0]:8.4f}, {error[1]:8.4f}]")

relative_error = np.abs((output_test - output_predicted) / (output_test + 1e-10))
print("Mean absolute error per output:", np.mean(absolute_difference, axis=0))
print("Mean relative error per output:", np.mean(relative_error, axis=0))

# Mean Absolute Error (MAE) for each output
mae_per_output = np.mean(np.abs(output_test - output_predicted), axis=0)

# Mean Relative Error for each output
# Add small epsilon to avoid division by zero
relative_error = np.abs((output_test - output_predicted) / (output_test + 1e-10))
mean_relative_error_per_output = np.mean(relative_error, axis=0)

# Display results
print("Mean Absolute Error per output:")
for i, mae in enumerate(mae_per_output):
    print(f"  Output {i+1}: {mae:.6f}")

print("\nMean Relative Error per output:")
for i, mre in enumerate(mean_relative_error_per_output):
    print(f"  Output {i+1}: {mre:.6f} ({mre*100:.2f}%)")

# Mean Absolute Percentage Error (MAPE) - often more interpretable
mape_per_output = np.mean(np.abs((output_test - output_predicted) / (output_test + 1e-10)), axis=0) * 100

print("\nMean Absolute Percentage Error (MAPE) per output:")
for i, mape in enumerate(mape_per_output):
    print(f"  Output {i+1}: {mape:.2f}%")

tf_model.summary()


# ==============================================================================
# TIME PREDICTION
# ==============================================================================


# Create a compiled prediction function
@tf.function(jit_compile=True)
def predict_xla(inputs):
    return tf_model(inputs, training=False)

# Get a single input sample (first row of your test data)
single_input = input_test_transformed[0:1]  # Keep it as (1, num_features) shape

# Warmup prediction (first prediction is often slower due to graph compilation)
# _ = tf_model.predict(single_input, batch_size=1, verbose=0)
_ = predict_xla(single_input)

# Time a single prediction
start_time = perf_counter()
# output_predicted = tf_model.predict(single_input, batch_size=1, verbose=0)
# _ = tf_model(single_input, training=False)
_ = predict_xla(single_input)
end_time = perf_counter()

inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Single inference time: {inference_time:.2f} ms")

# For more accurate timing, run multiple predictions and average
num_runs = 100
times = []

for i in range(num_runs):
    start_time = perf_counter()
    # _ = tf_model.predict(single_input, batch_size=1, verbose=0)
    # _ = tf_model(single_input, training=False)
    _ = predict_xla(single_input)
    end_time = perf_counter()
    times.append((end_time - start_time) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)
print(f"\nAverage inference time over {num_runs} runs: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"Min: {np.min(times):.2f} ms, Max: {np.max(times):.2f} ms")



