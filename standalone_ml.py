import numpy as np
import random
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

bounds1 = (-5.0, -10.0)
bounds2 = (10.0, 2.0)
# Generate input data: num_points evenly spaced points across each dimension
# np.column_stack combines two 1D arrays into a 2D array with 2 columns
# Each row represents a point (x1, x2) in the 2D domain

num_training_points = 100

inputs = np.column_stack(
    (
        np.linspace(bounds1[0], bounds1[1], num_training_points, dtype=np.single),
        np.linspace(bounds2[0], bounds2[1], num_training_points, dtype=np.single),
    )
)
# Create synthetic output data with 3 features based on mathematical functions
# This simulates a multi-output regression problem
outputs = np.column_stack(
    (
        0.8 * inputs[:, 0],
        0.5 * inputs[:, 1],
        np.sum(inputs, axis=1),
    )
)

# make testing data also. Don't want this on the same grid so make random points in the
#   domain
num_testing_points = 100
testing_inputs = np.column_stack(
    (
        np.linspace(bounds1[0], bounds1[1], num_testing_points, dtype=np.single),
        np.linspace(bounds2[0], bounds2[1], num_testing_points, dtype=np.single),
    )
)


# Create outputs for the testing inputs using the same functions as before
testing_outputs = np.column_stack(
    (
        0.8 * testing_inputs[:, 0],
        0.5 * testing_inputs[:, 1],
        np.sum(testing_inputs, axis=1),
    )
)

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
        tf.keras.layers.Dense(5, activation="tanh"),
        # Hidden layer 2: 5 neurons with tanh activation
        # Small network since we only need "smooth result" as comment mentions
        tf.keras.layers.Dense(5, activation="tanh"),
        # Output layer: 3 neurons (one for each output), no activation (linear)
        # Linear activation is typical for regression problems
        tf.keras.layers.Dense(n_outputs),
    ]
)
# Configure the model for training
tf_model.compile(
    # Adam optimizer with learning rate 0.01 - adaptive learning rate algorithm
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    # Mean Squared Error loss - standard for regression problems
    loss=tf.keras.losses.MeanSquaredError(),
)
# ==============================================================================
# MODEL TRAINING
# ==============================================================================
n_train = len(indices_training)

# Train the neural network
hist = tf_model.fit(
    # Training data (using original input, not transformed!)
    x=inputs[indices_training, :],   # Training inputs
    y=outputs[indices_training, :],  # Training targets
    epochs=400,           # Number of complete passes through training data
    batch_size=n_train,   # Use entire training set as one batch (batch gradient descent)
    verbose=0,            # Don't print training progress
    # Validation data for monitoring overfitting
    validation_data=(
        inputs[indices_validation, :],   # Validation inputs
        outputs[indices_validation, :],  # Validation targets
    ),
)

history = hist.history

# # ==============================================================================
# # PREDICTION AND INVERSE TRANSFORMATION
# # ==============================================================================

# test using the test data set
output_predicted_transformed = tf_model.predict(
    testing_inputs,            # Use preprocessed input data
    batch_size=len(testing_inputs),    # Process all samples at once
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


