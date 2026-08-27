import os
import random
import re
import uuid

import h5py
import numpy as np
import numpy.testing as npt
import openmdao.api as om
import pytest
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



# Build input and output.
domain = BoxDomain([-5.0, -10.0], [10.0, 2.0])
input = np.column_stack(
    (
        np.linspace(domain.min[0], domain.max[0], 1001, dtype=np.single),
        np.linspace(domain.min[1], domain.max[1], 1001, dtype=np.single),
    )
)
output = np.column_stack(
    (0.8 * input[:, 0] ** 2, 0.5 * input[:, 1] ** 2, np.sum(input**2, axis=1))
)
# Build input transformers.
sklearn_input_transformers = [
    StandardScaler(copy=True, with_mean=True, with_std=True),
    MinMaxScaler(feature_range=(-2.0, +2.0), copy=True, clip=False),
]
input_transformed = input.copy()
for transformer in sklearn_input_transformers:
    input_transformed = transformer.fit_transform(input_transformed)
si_input_transformers = [
    Transformer.from_sklearn(t) for t in sklearn_input_transformers
]
# Build output transformers.
sklearn_output_transformers = [
    StandardScaler(copy=True, with_mean=True, with_std=True),
    MinMaxScaler(feature_range=(-1.0, +1.0), copy=True, clip=False),
]
output_transformed = output.copy()
for transformer in sklearn_output_transformers:
    output_transformed = transformer.fit_transform(output_transformed)
si_output_transformers = [
    Transformer.from_sklearn(t) for t in sklearn_output_transformers
]
# Train Artificial Neural Network.
# We only need a smooth result, and therefore train a simple model for only a few epochs.
n_sample_total = input.shape[0]
n_inputs = input.shape[1]
n_outputs = output.shape[1]
ratio_train = 0.7
ratio_test = 0.2
ratio_validation = 1.0 - (ratio_train + ratio_test)
i_sample = np.arange(n_sample_total)
rng = np.random.default_rng(seed=12345)
rng.shuffle(i_sample)
n_train = round(ratio_train * n_sample_total)
n_validation = round(ratio_validation * n_sample_total)
all_points_parts = np.split(i_sample, (n_train, n_train + n_validation))
(
    sample_training_set,
    sample_validation_set,
    sample_testing_set,
) = all_points_parts
set_tf_seed()
tf_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(5, activation="tanh"),
        tf.keras.layers.Dense(5, activation="tanh"),
        tf.keras.layers.Dense(n_outputs),
    ]
)
tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.MeanSquaredError(),
)
hist = tf_model.fit(
    x=input[sample_training_set, :],
    y=output[sample_training_set, :],
    epochs=400,
    batch_size=n_train,
    verbose=0,
    validation_data=(
        input[sample_validation_set, :],
        output[sample_validation_set, :],
    ),
)
history = hist.history
output_predicted_transformed = tf_model.predict(
    input_transformed, batch_size=n_sample_total, verbose=0
)
output_predicted = output_predicted_transformed.copy()
for transformer in reversed(si_output_transformers):
    transformer.inverse_transform(output_predicted, inplace=True)
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
si_model = TensorFlowModel(
    tf_model,
    input_transformers=si_input_transformers,
    output_transformers=si_output_transformers,
    input_names=[f"x{i}" for i in range(n_inputs)],
    output_names=[f"y{i}" for i in range(n_outputs)],
    metadata={"Wöhler exponents": rng.uniform(0.0, 10.0, n_outputs)},
    domain=domain,
)

input_test = input[sample_testing_set, :]
output_test = output_predicted[sample_testing_set, :]



# Make the OpenMDAO problem.
component = SurrogateModelComp(
    model=si_model,
    # input_type=InputOutputType.JOINED,
    # output_type=InputOutputType.JOINED,
    input_type=InputOutputType.SPLIT,
    output_type=InputOutputType.SPLIT,
    n_points=input_test.shape[0],
)
problem = om.Problem()
problem.model.add_subsystem(
    # "surrogate", component, promotes_inputs=["x"], promotes_outputs=["y"]
    "surrogate", component, promotes_inputs=["*"], promotes_outputs=["*"]
)
problem.setup()

component.list_inputs()
component.list_outputs()


# Set the input.
problem.set_val("x", input_test)
# Evaluate the model.
problem.run_model()
# Check the output.
npt.assert_allclose(problem.get_val("y"), output_test, rtol=1e-6)

