import numpy as np
import openmdao.api as om

from surrogates_interface.domains import BoxDomain

from ml_surrogate_test import ml_surrogate_test

from kriging_surrogate_test import kriging_surrogate_test




def create_data_set():

    # Define a 2D rectangular domain for input data
    # This creates a box from (-5, -10) to (10, 2) in 2D space
    domain = BoxDomain([-5.0, -10.0], [10.0, 2.0])
    # Generate input data: num_points evenly spaced points across each dimension
    # np.column_stack combines two 1D arrays into a 2D array with 2 columns
    # Each row represents a point (x1, x2) in the 2D domain

    num_training_points = 100

    # TODO this just generates a line of points from bottom left to top right, which is not a grid!
    inputs = np.column_stack(
        (
            np.linspace(domain.min[0], domain.max[0], num_training_points, dtype=np.single),
            np.linspace(domain.min[1], domain.max[1], num_training_points, dtype=np.single),
        )
    )

    print(f"{inputs[0]=}")
    # Create synthetic output data with 3 features based on mathematical functions
    # This simulates a multi-output regression problem
    outputs = np.column_stack(
        (
            # Output 1: 0.8 * x1^2 (quadratic function of first input)
            0.8 * inputs[:, 0] ** 2,
            # Output 2: 0.5 * x2^2 (quadratic function of second input)
            0.5 * inputs[:, 1] ** 2,
            # Output 3: x1^2 + x2^2 (sum of squares - radial distance squared)
            np.sum(inputs**2, axis=1),
        )
    )

    print(f"{outputs[0]=}")

    # outputs = np.column_stack(
    #     (
    #         # Output 1: 0.8 * x1^2 (quadratic function of first input)
    #         0.8 * inputs[:, 0],
    #         # Output 2: 0.5 * x2^2 (quadratic function of second input)
    #         0.5 * inputs[:, 1],
    #         # Output 3: x1^2 + x2^2 (sum of squares - radial distance squared)
    #         np.sum(inputs, axis=1),
    #     )
    # )

    # make testing data also. Don't want this on the same grid so make random points in the
    #   domain
    # num_testing_points = 100
    # testing_inputs = np.column_stack(
    #     (
    #         np.random.uniform(domain.min[0], domain.max[0], num_testing_points),
    #         np.random.uniform(domain.min[1], domain.max[1], num_testing_points),
    #     )
    # )
    num_testing_points = 100
    testing_inputs = np.column_stack(
        (
            np.linspace(domain.min[0], domain.max[0], num_testing_points, dtype=np.single),
            np.linspace(domain.min[1], domain.max[1], num_testing_points, dtype=np.single),
        )
    )


    # Create outputs for the testing inputs using the same functions as before
    testing_outputs = np.column_stack(
        (
            0.8 * testing_inputs[:, 0] ** 2,
            0.5 * testing_inputs[:, 1] ** 2,
            np.sum(testing_inputs**2, axis=1),
        )
    )
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
    n_sample_total = inputs.shape[0]  # Total number of samples (1001)
    # n_inputs = inputs.shape[1]        # Number of input features (2)
    # n_outputs = outputs.shape[1]      # Number of output features (3)
    # Define train/validation/test split ratios
    ratio_train = 0.9       # 90% for training
    # ratio_test = 0.2        # 20% for testing
    ratio_validation = 1.0 - ratio_train  # 10% for validation

    # Create random split of data indices
    i_sample = np.arange(n_sample_total)  # [0, 1, 2, ..., 1000]
    rng = np.random.default_rng(seed=12345)  # Reproducible random number generator
    rng.shuffle(i_sample)  # Randomly shuffle the indices
    # Calculate actual number of samples for each set
    n_train = round(ratio_train * n_sample_total)          # ~700 samples
    n_validation = round(ratio_validation * n_sample_total) # ~100 samples
    # Test set gets the remainder (~200 samples)
    # Split shuffled indices into three sets
    all_points_parts = np.split(i_sample, [n_train,])
    (
        indices_training,    # Indices for training data
        indices_validation,  # Indices for validation data
    ) = all_points_parts

    return indices_training, indices_validation, testing_inputs, testing_outputs, inputs, outputs, domain


# def compute_mm_errors(surrogate_model, inputs, outputs):

#     prob = om.Problem()
#     prob.model.add_subsystem("surrogate_model", surrogate_model)
#     prob.setup()

#     # loop through all the rows of the inputs
#     error_sum = 0
#     for i in range(inputs.shape[0]):
#         # compute the output for each input
#         prob.set_val("surrogate_model.x", inputs[i])
#         prob.run_model()
#         # compare the results to the actual output
#         output_predicted = prob.get_val("surrogate_model.y")
#         output = outputs[i]

#         # compare the results to the actual output
#         # keep track of the average error

#         absolute_difference = np.absolute(output - output_predicted)
#         # keep track of the average error
#         error_sum += np.sum(absolute_difference)

#     # Compute the average error
#     average_error = error_sum / inputs.size
#     return average_error


# Create input data set and corresponding output using some function/component
indices_training, indices_validation, testing_inputs, testing_outputs, inputs, outputs, domain = create_data_set()

# loop through the surrogate models, training each of them using the input and output data
surrogate_model_tests = [
    ml_surrogate_test,
    kriging_surrogate_test,
]

for surrogate_test in surrogate_model_tests:
    model_name, error = surrogate_test(indices_training, indices_validation, testing_inputs, testing_outputs, inputs, outputs, domain)
    print(f"Surrogate Model: {model_name}, Mean Absolute Error: {error:.4f}")
