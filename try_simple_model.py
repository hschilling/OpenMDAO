import random
import os
import numpy as np
import numpy.testing as npt
import openmdao.api as om
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from surrogates_interface.domains import BoxDomain
from surrogates_interface.openmdao import (
    InputOutputType,
    SurrogateModelComp,
    SurrogateModelCompMatrixFree,
)
from surrogates_interface.surrogates import (
    H5_STR,
    ADMode,
    SurrogateModel,
    TensorFlowModel,
)
from surrogates_interface.transformers import MinMaxScaler as MinMaxScalerSI
from surrogates_interface.transformers import Transformer


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



def make_data_mimo():
    # ==============================================================================
    # SYNTHETIC DATA GENERATION
    # ==============================================================================

    # Define a 2D rectangular domain for input data
    # This creates a box from (-5, -10) to (10, 2) in 2D space
    domain = BoxDomain([-5.0, -10.0], [10.0, 2.0])
    # Generate input data: 1001 evenly spaced points across each dimension
    # np.column_stack combines two 1D arrays into a 2D array with 2 columns
    # Each row represents a point (x1, x2) in the 2D domain
    input = np.column_stack(
        (
            np.linspace(domain.min[0], domain.max[0], 1001, dtype=np.single),
            np.linspace(domain.min[1], domain.max[1], 1001, dtype=np.single),
        )
    )
    # Create synthetic output data with 3 features based on mathematical functions
    # This simulates a multi-output regression problem
    output = np.column_stack(
        (
            # Output 1: 0.8 * x1^2 (quadratic function of first input)
            0.8 * input[:, 0] ** 2, 
            # Output 2: 0.5 * x2^2 (quadratic function of second input)
            0.5 * input[:, 1] ** 2, 
            # Output 3: x1^2 + x2^2 (sum of squares - radial distance squared)
            np.sum(input**2, axis=1))
    )
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
    input_transformed = input.copy() # Start with original data
    for transformer in sklearn_input_transformers:
        # Each transformer learns parameters from data and transforms it
        input_transformed = transformer.fit_transform(input_transformed)
    # Convert sklearn transformers to a custom format (likely for inference/deployment)
    # This preserves the transformation parameters for later use
    si_input_transformers = [
        Transformer.from_sklearn(t) for t in sklearn_input_transformers
    ]
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
    output_transformed = output.copy()
    for transformer in sklearn_output_transformers:
        output_transformed = transformer.fit_transform(output_transformed)
    # Convert to custom transformer format for consistency
    si_output_transformers = [
        Transformer.from_sklearn(t) for t in sklearn_output_transformers
    ]
    # Train Artificial Neural Network.
    # We only need a smooth result, and therefore train a simple model for only a few epochs.
    # ==============================================================================
    # DATA SPLITTING FOR MACHINE LEARNING
    # ==============================================================================

    # Get dataset dimensions
    n_sample_total = input.shape[0]  # Total number of samples (1001)
    n_inputs = input.shape[1]        # Number of input features (2)
    n_outputs = output.shape[1]      # Number of output features (3)
    # Define train/validation/test split ratios
    ratio_train = 0.7       # 70% for training
    ratio_test = 0.2        # 20% for testing
    ratio_validation = 1.0 - (ratio_train + ratio_test)  # 10% for validation

    # Create random split of data indices
    i_sample = np.arange(n_sample_total)  # [0, 1, 2, ..., 1000]
    rng = np.random.default_rng(seed=12345)  # Reproducible random number generator
    rng.shuffle(i_sample)  # Randomly shuffle the indices
    # Calculate actual number of samples for each set
    n_train = round(ratio_train * n_sample_total)          # ~700 samples
    n_validation = round(ratio_validation * n_sample_total) # ~100 samples
    # Test set gets the remainder (~200 samples)
    # Split shuffled indices into three sets
    all_points_parts = np.split(i_sample, (n_train, n_train + n_validation))
    (
        sample_training_set,    # Indices for training data
        sample_validation_set,  # Indices for validation data 
        sample_testing_set,     # Indices for test data
    ) = all_points_parts
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

    # Train the neural network
    hist = tf_model.fit(
        # Training data (using original input, not transformed!)
        x=input[sample_training_set, :],   # Training inputs
        y=output[sample_training_set, :],  # Training targets
        epochs=400,           # Number of complete passes through training data
        batch_size=n_train,   # Use entire training set as one batch (batch gradient descent)
        verbose=0,            # Don't print training progress
        # Validation data for monitoring overfitting
        validation_data=(
            input[sample_validation_set, :],   # Validation inputs
            output[sample_validation_set, :],  # Validation targets
        ),
    )

    history = hist.history

    # ==============================================================================
    # PREDICTION AND INVERSE TRANSFORMATION
    # ==============================================================================

    # Make predictions using the transformed input data
    # Note: This uses input_transformed (preprocessed) for prediction
    output_predicted_transformed = tf_model.predict(
        input_transformed,            # Use preprocessed input data
        batch_size=n_sample_total,    # Process all samples at once
        verbose=0                     # Don't print prediction progress
    )

    # Convert predictions back to original output scale
    # Start with the transformed predictions
    output_predicted = output_predicted_transformed.copy()

    # Apply inverse transformations in reverse order to get back to original scale
    for transformer in reversed(si_output_transformers):
        # Each transformer undoes its transformation (e.g., unscale, unstandardize)
        transformer.inverse_transform(output_predicted, inplace=True)
            
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
    if False:
        import matplotlib.pyplot as plt

        # Plot history.
        fig_name = "training_history"
        fig, ax = plt.subplots(num=fig_name, dpi=300)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error [-]")
        ax.set_yscale("log")
        ax.grid(True)
        ax.plot(history["loss"], label="Training")
        ax.plot(history["val_loss"], label="Validation")
        ax.legend(loc="upper right")
        # Plot 1 to 1.
        fig_name = "one_to_one"
        fig, ax = plt.subplots(num=fig_name, dpi=300)
        ax.set_xlabel("Exact Output")
        ax.set_ylabel("Predicted Output")
        ax.grid(True)
        ax.axline((0.0, 0.0), slope=1.0, color="k", linestyle="--")
        ax.scatter(output.ravel(), output_predicted.ravel())
        
        plt.show()
    si_model = TensorFlowModel(
        tf_model,
        input_transformers=si_input_transformers,
        output_transformers=si_output_transformers,
        input_names=[f"x{i}" for i in range(n_inputs)],
        output_names=[f"y{i}" for i in range(n_outputs)],
        metadata={"Wöhler exponents": rng.uniform(0.0, 10.0, n_outputs)},
        domain=domain,
    )
    return (
        si_model,
        input[sample_testing_set, :],
        output_predicted[sample_testing_set, :],
    )



si_model, input_test, output_test = make_data_mimo()
# Make the OpenMDAO problem.
component = SurrogateModelComp(
    model=si_model,
    input_type=InputOutputType.JOINED,
    output_type=InputOutputType.JOINED,
    n_points=input_test.shape[0],
)
problem = om.Problem()
problem.model.add_subsystem(
    "surrogate", component, promotes_inputs=["x"], promotes_outputs=["y"]
)
problem.setup()
# Set the input.
problem.set_val("x", input_test)
# Evaluate the model.
problem.run_model()

# print(problem["x"])
# Check the output.
npt.assert_allclose(problem.get_val("y"), output_test, rtol=1e-6)

