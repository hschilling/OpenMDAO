import numpy as np
import random
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import openmdao.api as om

# ==============================================================================
# SUMMARY
# ==============================================================================
# This code:
# 1. Creates synthetic 2D input data and 3D output data with known relationships
# 2. Preprocesses both input and output data using standardization and scaling
# 3. Splits data into train/validation/test sets randomly
# 4. Trains a small neural network (2→5→5→3) for 400 epochs
# 5. Makes predictions and converts them back to original output scale
#
# Key observations:
# - Training uses ORIGINAL input/output data (not transformed)
# - Prediction uses TRANSFORMED input data
# - Final predictions are inverse-transformed back to original output scale
# - This is a multi-output regression problem (predicting 3 values from 2 inputs)

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
# generate synthetic data FOR testing MACHINE LEARNING code
# ==============================================================================

# Define the rectangle bounds
x_min, x_max = -5, 10   # X dimension bounds
y_min, y_max = -10, 2   # Y dimension bounds
# Generate input data: num_points evenly spaced points across each dimension
# np.column_stack combines two 1D arrays into a 2D array with 2 columns
# Each row represents a point (x1, x2) in the 2D domain

# num_training_points = 100

num_points_along_dim = 33
# Create 1D arrays for each dimension
x = np.linspace(x_min, x_max, num_points_along_dim)  # 10 points in X
y = np.linspace(y_min, y_max, num_points_along_dim)  # 25 points in Y

# Create a meshgrid
X, Y = np.meshgrid(x, y)

inputs = np.column_stack([X.ravel(), Y.ravel()])


# inputs = np.column_stack(
#     (
#         np.linspace(-5, 10, num_points_along_dim, dtype=np.single),
#         np.linspace(-10, 2, num_points_along_dim, dtype=np.single),
#     )
# )
# Create synthetic output data with 3 features based on mathematical functions
# This simulates a multi-output regression problem
# outputs = np.column_stack(
#     (
#         0.8 * inputs[:, 0],
#         0.5 * inputs[:, 1],
#         np.sum(inputs, axis=1),
#     )
# )

outputs = np.column_stack(
    (
        # Output 1: 0.8 * x1^2 (quadratic function of first input)
        # 0.8 * inputs[:, 0] ** 2,
        0.8 * inputs[:, 0] ,
        # Output 2: 0.5 * x2^2 (quadratic function of second input)
        0.5 * inputs[:, 1] ** 2,
        # Output 3: x1^2 + x2^2 (sum of squares - radial distance squared)
        np.sum(inputs**2, axis=1),
    )
)


# make testing data also. Don't want this on the same grid so make random points in the
#   domain
# num_testing_points = 100
# testing_inputs = np.column_stack(
#     (
#         np.linspace(-5, 10, num_testing_points, dtype=np.single),
#         np.linspace(-10, 2, num_testing_points, dtype=np.single),
#     )
# )


# Create 1D arrays for each dimension
x = np.random.uniform(x_min, x_max, num_points_along_dim)  # 10 points in X
y = np.random.uniform(y_min, y_max, num_points_along_dim)

# Create a meshgrid
X, Y = np.meshgrid(x, y)

# Stack and reshape to (250, 2)
testing_inputs = np.column_stack([X.ravel(), Y.ravel()])

testing_outputs = np.column_stack(
    (
        # 0.8 * testing_inputs[:, 0] ** 2,
        0.8 * testing_inputs[:, 0],
        0.5 * testing_inputs[:, 1] ** 2,
        np.sum(testing_inputs**2, axis=1),
    )
)


# Create outputs for the testing inputs using the same functions as before
# testing_outputs = np.column_stack(
#     (
#         0.8 * testing_inputs[:, 0],
#         0.5 * testing_inputs[:, 1],
#         np.sum(testing_inputs, axis=1),
#     )
# )

# ==============================================================================
# DATA SPLITTING FOR MACHINE LEARNING
# ==============================================================================

# Get dataset dimensions
n_sample_total = inputs.shape[0]  # Total number of samples
n_inputs = inputs.shape[1]        # Number of input features (2)
n_outputs = outputs.shape[1]      # Number of output features (3)
# Define train/validation/test split ratios
ratio_train = 0.9       # 90% for training
ratio_validation = 1.0 - ratio_train  # 10% for validation

# Create random split of data indices
i_sample = np.arange(n_sample_total)  # [0, 1, 2, ..., 1000]
rng = np.random.default_rng(seed=12345)  # Reproducible random number generator
rng.shuffle(i_sample)  # Randomly shuffle the indices
# Calculate actual number of samples for each set
n_train = round(ratio_train * n_sample_total)
n_validation = round(ratio_validation * n_sample_total)
# Split shuffled indices into three sets
all_points_parts = np.split(i_sample, [n_train,])
(
    indices_training,    # Indices for training data
    indices_validation,  # Indices for validation data
) = all_points_parts

# ==============================================================================
# INPUT DATA PREPROCESSING
# ==============================================================================

# Define a chain of sklearn preprocessing transformers for input data
sklearn_input_transformers = [
    # Step 1: Standardize features (mean=0, std=1) - Z-score normalization
    StandardScaler(copy=True, with_mean=True, with_std=True),
    # Step 2: Scale to range [-2, +2] after standardization
    MinMaxScaler(feature_range=(-2.0, +2.0), copy=True, clip=False),
]
# Apply the transformation pipeline to input data
input_transformed = inputs.copy() # Start with original data
for transformer in sklearn_input_transformers:
    # Each transformer learns parameters from data and transforms it
    input_transformed = transformer.fit_transform(input_transformed)

# ==============================================================================
# OUTPUT DATA PREPROCESSING
# ==============================================================================

# Define preprocessing transformers for output data (targets)
sklearn_output_transformers = [
    # Step 1: Standardize outputs (mean=0, std=1)
    StandardScaler(copy=True, with_mean=True, with_std=True),
    # Step 2: Scale to range [-1, +1] - smaller range than inputs
    MinMaxScaler(feature_range=(-1.0, +1.0), copy=True, clip=False),
]
# Apply transformation pipeline to output data
output_transformed = outputs.copy()
for transformer in sklearn_output_transformers:
    output_transformed = transformer.fit_transform(output_transformed)

# Train Artificial Neural Network.
# We only need a smooth result, and therefore train a simple model for only a few epochs.
# ==============================================================================
# DATA SPLITTING FOR MACHINE LEARNING
# ==============================================================================

# Get dataset dimensions
n_sample_total = inputs.shape[0]  # Total number of samples (1001)
n_inputs = inputs.shape[1]        # Number of input features (2)
n_outputs = outputs.shape[1]      # Number of output features (3)

# ==============================================================================
# NEURAL NETWORK MODEL DEFINITION
# ==============================================================================
# Set TensorFlow random seed for reproducibility (custom function)
set_tf_seed()
# Create a Sequential neural network model
tf_model = tf.keras.Sequential(
    [
        # Input layer: explicitly define input shape (2 features)
        tf.keras.Input(shape=(n_inputs,)),
        # Hidden layer 1: 5 neurons with tanh activation
        # tanh outputs values in range [-1, 1], good for normalized data
        tf.keras.layers.Dense(50, activation="relu",
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Hidden layer 2: 5 neurons with tanh activation
        # Small network since we only need "smooth result" as comment mentions
        tf.keras.layers.Dense(50, activation="relu",
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Output layer: 3 neurons (one for each output), no activation (linear)
        # Linear activation is typical for regression problems
        tf.keras.layers.Dense(n_outputs),
    ]
)
# Configure the model for training
tf_model.compile(
    # Adam optimizer with learning rate 0.01 - adaptive learning rate algorithm
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
    # Mean Squared Error loss - standard for regression problems
    loss=tf.keras.losses.MeanSquaredError(),
)
# ==============================================================================
# MODEL TRAINING
# ==============================================================================
n_train = len(indices_training)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=1e-6)

hist = tf_model.fit(
    x=input_transformed[indices_training, :],
    y=output_transformed[indices_training, :],
    # epochs=5000,
    epochs=5,
    batch_size=32,
    verbose=0,
    validation_data=(input_transformed[indices_validation, :],
                     output_transformed[indices_validation, :]),
    callbacks=[reduce_lr],
)


# Train the neural network
# hist = tf_model.fit(
#     # Training data (using original input, not transformed!)
#     x=input_transformed[indices_training, :],   # Training inputs
#     y=output_transformed[indices_training, :],  # Training targets
#     epochs=2000,           # Number of complete passes through training data
#     batch_size=32,   # Use entire training set as one batch (batch gradient descent)
#     verbose=0,            # Don't print training progress
#     # Validation data for monitoring overfitting
#     validation_data=(
#         input_transformed[indices_validation, :],   # Validation inputs
#         output_transformed[indices_validation, :],  # Validation targets
#     ),
# )

history = hist.history


print(f"Final training loss: {history['loss'][-1]:.6f}")
print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

# Plot training history
# import matplotlib.pyplot as plt
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.yscale('log')
# plt.show()


# # ==============================================================================
# # PREDICTION AND INVERSE TRANSFORMATION
# # ==============================================================================


# ==============================================================================
# TRANSFORM TESTING DATA
# ==============================================================================
testing_inputs_transformed = testing_inputs.copy()
for transformer in sklearn_input_transformers:
    testing_inputs_transformed = transformer.transform(testing_inputs_transformed)

# test using the test data set
output_predicted_transformed = tf_model.predict(
    testing_inputs_transformed,            # Use preprocessed input data
    batch_size=len(testing_inputs_transformed),    # Process all samples at once
    verbose=0                     # Don't print prediction progress
)
# Convert predictions back to original output scale
# Start with the transformed predictions
output_predicted = output_predicted_transformed.copy()

# Apply inverse transformations in reverse order to get back to original scale
for transformer in reversed(sklearn_output_transformers):
    # Each transformer undoes its transformation (e.g., unscale, unstandardize)
    # transformer.inverse_transform(output_predicted, inplace=True)
    output_predicted = transformer.inverse_transform(output_predicted)

# testing_outputs = outputs[indices_testing, :]
absolute_difference = np.absolute(testing_outputs - output_predicted)
# keep track of the average error
error = np.sum(absolute_difference)


print(f"{testing_outputs[0:2]=}")
print(f"{output_predicted[0:2]=}")

if False:
    print("Row | Actual                          | Predicted                       | Error")
    print("-" * 85)
    for i in range(testing_outputs.shape[0]):
        actual = testing_outputs[i]
        predicted = output_predicted[i]
        error = np.abs(actual - predicted)
        print(f"{i:3d} | [{actual[0]:8.4f}, {actual[1]:8.4f}, {actual[2]:8.4f}] | "
            f"[{predicted[0]:8.4f}, {predicted[1]:8.4f}, {predicted[2]:8.4f}] | "
            f"[{error[0]:8.4f}, {error[1]:8.4f}, {error[2]:8.4f}]")

relative_error = np.abs((testing_outputs - output_predicted) / (testing_outputs + 1e-10))
print("Mean absolute error per output:", np.mean(absolute_difference, axis=0))
print("Mean relative error per output:", np.mean(relative_error, axis=0))

# Mean Absolute Error (MAE) for each output
mae_per_output = np.mean(np.abs(testing_outputs - output_predicted), axis=0)

# Mean Relative Error for each output
# Add small epsilon to avoid division by zero
relative_error = np.abs((testing_outputs - output_predicted) / (testing_outputs + 1e-10))
mean_relative_error_per_output = np.mean(relative_error, axis=0)

# Display results
print("Mean Absolute Error per output:")
for i, mae in enumerate(mae_per_output):
    print(f"  Output {i+1}: {mae:.6f}")

print("\nMean Relative Error per output:")
for i, mre in enumerate(mean_relative_error_per_output):
    print(f"  Output {i+1}: {mre:.6f} ({mre*100:.2f}%)")

# Mean Absolute Percentage Error (MAPE) - often more interpretable
mape_per_output = np.mean(np.abs((testing_outputs - output_predicted) / (testing_outputs + 1e-10)), axis=0) * 100

print("\nMean Absolute Percentage Error (MAPE) per output:")
for i, mape in enumerate(mape_per_output):
    print(f"  Output {i+1}: {mape:.2f}%")

# Train a simple linear regression model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(input_transformed[indices_training, :],
                 output_transformed[indices_training, :])

# Predict
output_predicted_transformed = linear_model.predict(testing_inputs_transformed)
output_predicted = output_predicted_transformed.copy()

# Apply inverse transformations in reverse order to get back to original scale
for transformer in reversed(sklearn_output_transformers):
    # Each transformer undoes its transformation (e.g., unscale, unstandardize)
    # transformer.inverse_transform(output_predicted, inplace=True)
    output_predicted = transformer.inverse_transform(output_predicted)

relative_error = np.abs((testing_outputs - output_predicted) / (testing_outputs + 1e-10))
print("linear regression: Mean relative error per output:", np.mean(relative_error, axis=0))

# Mean Absolute Percentage Error (MAPE) - often more interpretable
mape_per_output = np.mean(np.abs((testing_outputs - output_predicted) / (testing_outputs + 1e-10)), axis=0) * 100

print("\nLinear regression: Mean Absolute Percentage Error (MAPE) per output:")
for i, mape in enumerate(mape_per_output):
    print(f"  Output {i+1}: {mape:.2f}%")



from surrogates_interface.openmdao import (
    InputOutputType,
    SurrogateModelComp,
)
from surrogates_interface.surrogates import (
    # H5_STR,
    # ADMode,
    # SurrogateModel,
    TensorFlowModel,
)
from surrogates_interface.transformers import Transformer
from surrogates_interface.domains import BoxDomain

si_input_transformers = [
    Transformer.from_sklearn(t) for t in sklearn_input_transformers
]
si_output_transformers = [
    Transformer.from_sklearn(t) for t in sklearn_output_transformers
]

domain = BoxDomain([-5.0, -10.0], [10.0, 2.0])

si_model = TensorFlowModel(
    tf_model,
    input_transformers=si_input_transformers,
    output_transformers=si_output_transformers,
    input_names=[f"x{i}" for i in range(n_inputs)],
    output_names=[f"y{i}" for i in range(n_outputs)],
    metadata={"Wöhler exponents": rng.uniform(0.0, 10.0, n_outputs)},
    domain=domain,
)
# input is an array of shape 1001,2 in the example. So num points by vars per point
# input_test = inputs[indices_testing, :]
# output_test = outputs_predicted[sample_testing_set, :],


# Make the OpenMDAO problem.
component = SurrogateModelComp(
    model=si_model,
    input_type=InputOutputType.SPLIT,
    output_type=InputOutputType.SPLIT,
    n_points=1,
)

# build the model
prob = om.Problem()

prob.model.add_subsystem('tf_comp', component, promotes_inputs=['x0', 'x1'])

prob.model.add_design_var('x0', lower=-5, upper=10)
prob.model.add_design_var('x1', lower=-10, upper=2)
prob.model.add_objective('tf_comp.y2')

prob.setup()
prob.final_setup()

# Set initial values.
prob.set_val('x0', 3.0)
prob.set_val('x1', -4.0)

# run the design optimization
prob.run_driver()

print(f"x0 = {prob.get_val('x0')}")
print(f"x1 = {prob.get_val('x1')}")
print(f"y2 = {prob.get_val('tf_comp.y2')}")
