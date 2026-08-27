# -*- coding: utf-8 -*-
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


# %% Test TensorFlowModel.


def test_init_n_features():
    obj = TensorFlowModel(
        tf.keras.Sequential(),
        input_names=("x1", "x2"),
        output_names=("y1",),
    )
    assert obj.n_inputs == 2
    assert obj.n_outputs == 1


def test_init_channel_names():
    obj = TensorFlowModel(tf.keras.Sequential(), n_inputs=2, n_outputs=1)
    assert obj.input_names == ["x_0", "x_1"]
    assert obj.output_names == ["y_0"]


def test_init_features_in_mismatch():
    with pytest.raises(AssertionError):
        TensorFlowModel(tf.keras.Sequential(), input_names=("x1", "x2"), n_inputs=5)


def test_init_features_out_mismatch():
    with pytest.raises(AssertionError):
        TensorFlowModel(tf.keras.Sequential(), output_names=("y1",), n_outputs=4)


def test_surrogate_model_jvp():
    obj = SurrogateModel(lambda x: x)
    with pytest.raises(NotImplementedError):
        obj._predict_model_jvp(np.array([0.0]), np.array([0.0]))


def test_surrogate_model_vjp():
    obj = SurrogateModel(lambda x: x)
    with pytest.raises(NotImplementedError):
        obj._predict_model_vjp(np.array([0.0]), np.array([0.0]))


def test_init_features_inout_match():
    obj = TensorFlowModel(
        tf.keras.Sequential(),
        input_names=("x1", "x2"),
        n_inputs=2,
        output_names=("y1",),
        n_outputs=1,
    )
    assert obj.input_names == ("x1", "x2")
    assert obj.output_names == ("y1",)
    assert obj.n_inputs == 2
    assert obj.n_outputs == 1


# These tests are meant to cover the following cases:
#   - Single Input Single Output
#   - Multiple Input Single Output
#   - Single Input Multiple Output
#   - Multiple Input Multiple Output
# In all cases, the input and output transformers are included.
# The tests focus on predict_output() and predict_jacobian(),
# where the latter is checked with finite differences against the model output.


@pytest.fixture
def make_data_siso():
    # Build input and output.
    domain = BoxDomain([-5.0], [10.0])
    input = np.linspace(domain.min[0], domain.max[0], 1001, dtype=np.single).reshape(
        -1, 1
    )
    output = input**2
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
        # Plot function.
        fig_name = "function"
        fig, ax = plt.subplots(num=fig_name, dpi=300)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.plot(input.ravel(), output.ravel(), label="Exact")
        ax.plot(input.ravel(), output_predicted.ravel(), label="ANN")
        ax.legend()
    si_model = TensorFlowModel(
        tf_model,
        si_input_transformers,
        si_output_transformers,
        n_inputs=1,
        n_outputs=1,
        domain=domain,
    )
    return (
        si_model,
        input[sample_testing_set, :],
        output_predicted[sample_testing_set, :],
    )


def test_predict_output_siso(make_data_siso):
    si_model, input, output = make_data_siso
    output_si = si_model.predict_output(input)
    npt.assert_array_almost_equal_nulp(output_si, output, nulp=10)


def test_predict_jacobian_siso(make_data_siso):
    si_model, input, _ = make_data_siso
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input)
    # Compute jacobian using central finite differences.
    h = 1e-3
    jacobian_fd = np.zeros(jacobian_si.shape)
    jacobian_fd[:, 0, 0] = (
        si_model.predict_output(input + h).ravel()
        - si_model.predict_output(input - h).ravel()
    ) / (2.0 * h)
    if False:
        # Sensitivity study on the step size.
        import matplotlib.pyplot as plt

        fig_name = "jacobian"
        fig, ax = plt.subplots(num=fig_name, dpi=300)
        ax.set_xlabel("x")
        ax.set_ylabel("f'(x)")
        ax.grid(True)
        input = np.sort(input, axis=0)
        # Compute central finite differences.
        for h in [1e-3, 1e-2, 1e-1]:
            yl = si_model.predict_output(input - h)
            yr = si_model.predict_output(input + h)
            jacobian_fd = (yr - yl) / (2.0 * h)
            ax.plot(input.ravel(), jacobian_fd.ravel(), label=f"Central, h = {h:.0e}")
        # Compute forward finite differences.
        yc = si_model.predict_output(input)
        for h in [1e-3, 1e-2, 1e-1]:
            yr = si_model.predict_output(input + h)
            jacobian_fd = (yr - yc) / h
            ax.plot(
                input.ravel(),
                jacobian_fd.ravel(),
                "--",
                label=f"Forward, h = {h:.0e}",
            )
        ax.legend()
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


@pytest.fixture
def make_data_miso():
    # Build input and output.
    domain = BoxDomain([-5.0, -10.0], [10.0, 2.0])
    input = np.column_stack(
        (
            np.linspace(domain.min[0], domain.max[0], 1001, dtype=np.single),
            np.linspace(domain.min[1], domain.max[1], 1001, dtype=np.single),
        )
    )
    output = np.sum(input**2, axis=1, keepdims=True)
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
        si_input_transformers,
        si_output_transformers,
        n_inputs=2,
        n_outputs=1,
        domain=domain,
    )
    return (
        si_model,
        input[sample_testing_set, :],
        output_predicted[sample_testing_set, :],
    )


def test_predict_output_miso(make_data_miso):
    si_model, input, output = make_data_miso
    output_si = si_model.predict_output(input)
    npt.assert_array_almost_equal_nulp(output_si, output, nulp=10)


def test_predict_output_outside_of_domain_miso(make_data_miso):
    si_model, _, _ = make_data_miso
    input = np.column_stack(
        (
            np.linspace(-6.0, 20.0, 4, dtype=np.single),
            np.linspace(-2.0, 1.0, 4, dtype=np.single),
        )
    )
    with pytest.warns(
        UserWarning, match="75.0% of points are outside of the input domain."
    ):
        si_model.predict_output(input)


def test_predict_jacobian_miso(make_data_miso):
    si_model, input, _ = make_data_miso
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input)
    # Compute jacobian using central finite differences.
    jacobian_fd = np.zeros(jacobian_si.shape)
    h = 1e-3
    for i in range(input.shape[1]):
        xm = input.copy()
        xp = input.copy()
        xm[:, i] -= h
        xp[:, i] += h
        jacobian_fd[:, 0, i] = (
            si_model.predict_output(xp).ravel() - si_model.predict_output(xm).ravel()
        )
    jacobian_fd /= 2.0 * h
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


def test_predict_jacobian_outside_of_domain_miso(make_data_miso):
    si_model, _, _ = make_data_miso
    input = np.column_stack(
        (
            np.linspace(-6.0, 20.0, 4, dtype=np.single),
            np.linspace(-2.0, 1.0, 4, dtype=np.single),
        )
    )
    with pytest.warns(
        UserWarning, match="75.0% of points are outside of the input domain."
    ):
        si_model.predict_jacobian(input)


@pytest.fixture
def make_data_simo():
    # Build input and output.
    domain = BoxDomain([-5.0], [10.0])
    input = np.linspace(domain.min[0], domain.max[0], 1001, dtype=np.single).reshape(
        -1, 1
    )
    output = np.column_stack((input**2, 0.5 * input**2))
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
    output_predicted_transformed = tf_model(input_transformed, training=False).numpy()
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
        # Plot function.
        fig_name = "function"
        fig, ax = plt.subplots(num=fig_name, dpi=300)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        for i in range(n_outputs):
            ax.plot(input.ravel(), output[:, i], color=f"C{i}", label=f"Exact {i}")
            ax.plot(
                input.ravel(),
                output_predicted[:, i].ravel(),
                "--",
                color=f"C{i}",
                label=f"ANN {i}",
            )
        ax.legend()
    si_model = TensorFlowModel(
        tf_model,
        si_input_transformers,
        si_output_transformers,
        n_inputs=1,
        n_outputs=2,
        domain=domain,
    )
    return (
        si_model,
        input[sample_testing_set, :],
        output_predicted[sample_testing_set, :],
    )


def test_predict_output_simo(make_data_simo):
    si_model, input, output = make_data_simo
    output_si = si_model.predict_output(input)
    npt.assert_array_almost_equal_nulp(output_si, output, nulp=40)


def test_predict_jacobian_simo(make_data_simo):
    si_model, input, _ = make_data_simo
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input)
    # Compute jacobian using central finite differences.
    h = 1e-3
    jacobian_fd = np.zeros(jacobian_si.shape)
    jacobian_fd[:, :, 0] = (
        si_model.predict_output(input + h) - si_model.predict_output(input - h)
    ) / (2.0 * h)
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


@pytest.fixture
def make_data_mimo():
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
    return (
        si_model,
        input[sample_testing_set, :],
        output_predicted[sample_testing_set, :],
    )


def test_predict_output_mimo(make_data_mimo):
    si_model, input, output = make_data_mimo
    output_si = si_model.predict_output(input)
    npt.assert_array_almost_equal_nulp(output_si, output, nulp=10)


def test_predict_jacobian_mimo_automatic(make_data_mimo):
    # The AD mode is chosen automatically based on the number of inputs and outputs.
    si_model, input, _ = make_data_mimo
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input)
    # Compute jacobian using central finite differences.
    jacobian_fd = np.zeros(jacobian_si.shape)
    h = 1e-3
    for i in range(input.shape[1]):
        xm = input.copy()
        xp = input.copy()
        xm[:, i] -= h
        xp[:, i] += h
        jacobian_fd[:, :, i] = si_model.predict_output(xp) - si_model.predict_output(xm)
    jacobian_fd /= 2.0 * h
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


def test_predict_jacobian_mimo_fwd(make_data_mimo):
    # Compute the Jacobian using forward mode AD.
    si_model, input, _ = make_data_mimo
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input, ad_mode=ADMode.FWD)
    # Compute jacobian using central finite differences.
    jacobian_fd = np.zeros(jacobian_si.shape)
    h = 1e-3
    for i in range(input.shape[1]):
        xm = input.copy()
        xp = input.copy()
        xm[:, i] -= h
        xp[:, i] += h
        jacobian_fd[:, :, i] = si_model.predict_output(xp) - si_model.predict_output(xm)
    jacobian_fd /= 2.0 * h
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


def test_predict_jacobian_mimo_rev(make_data_mimo):
    # Compute the Jacobian using reverse mode AD.
    si_model, input, _ = make_data_mimo
    # Compute jacobian using the chain rule and TensorFlow.
    jacobian_si = si_model.predict_jacobian(input, ad_mode=ADMode.REV)
    # Compute jacobian using central finite differences.
    jacobian_fd = np.zeros(jacobian_si.shape)
    h = 1e-3
    for i in range(input.shape[1]):
        xm = input.copy()
        xp = input.copy()
        xm[:, i] -= h
        xp[:, i] += h
        jacobian_fd[:, :, i] = si_model.predict_output(xp) - si_model.predict_output(xm)
    jacobian_fd /= 2.0 * h
    npt.assert_allclose(jacobian_si, jacobian_fd, rtol=1e-1, atol=1e-1)


def test_save_load_h5(make_data_mimo, tmp_model_data_paths):
    si_model, _, _ = make_data_mimo
    model_path, extra_data_path = tmp_model_data_paths
    si_model.save_h5(model_path, extra_data_path)
    si_model_loaded = TensorFlowModel.load_h5(model_path, extra_data_path)
    assert si_model == si_model_loaded


def test_save_load_h5_without_domain(make_data_mimo, tmp_model_data_paths):
    model, _, _ = make_data_mimo
    model.domain = None
    model_path, extra_data_path = tmp_model_data_paths
    model.save_h5(model_path, extra_data_path)
    model_loaded = TensorFlowModel.load_h5(model_path, extra_data_path)
    assert model == model_loaded


def test_save_load_h5_without_n_in_out(make_data_mimo, tmp_model_data_paths):
    model, _, _ = make_data_mimo
    # model.n_inputs = None
    # model.n_outputs = None
    model_path, extra_data_path = tmp_model_data_paths
    model.save_h5(model_path, extra_data_path)
    # Delete n_inputs and n_outputs from the saved file.
    # They will be reconstructed during loading from the input and output names.
    with h5py.File(extra_data_path, "a") as fid:
        del fid["n_inputs"]
        del fid["n_outputs"]
    model_loaded = TensorFlowModel.load_h5(model_path, extra_data_path)
    assert model == model_loaded


def test_save_load_h5_without_n_in_out_and_names(make_data_mimo, tmp_model_data_paths):
    model, _, _ = make_data_mimo
    # model.n_inputs = None
    # model.n_outputs = None
    model_path, extra_data_path = tmp_model_data_paths
    model.save_h5(model_path, extra_data_path)
    # Delete n_inputs and n_outputs from the saved file.
    # Delete also the input and output names.
    with h5py.File(extra_data_path, "a") as fid:
        del fid["n_inputs"]
        del fid["n_outputs"]
        del fid["input_names"]
        del fid["output_names"]
        fid.create_dataset("input_names", data=[], dtype=H5_STR)
        fid.create_dataset("output_names", data=[], dtype=H5_STR)
    with pytest.warns(UserWarning):
        TensorFlowModel.load_h5(model_path, extra_data_path)


def test_eq_wrong_metadata(make_data_siso, make_data_mimo):
    # This test will fail because of the different metadata.
    model_siso, _, _ = make_data_siso
    model_mimo, _, _ = make_data_mimo
    with pytest.raises(AssertionError):
        assert model_siso == model_mimo


def test_eq_missing_domain(make_data_siso):
    model_siso, _, _ = make_data_siso
    obj = SurrogateModel(lambda x: x + 2.0)
    with pytest.raises(AssertionError):
        assert model_siso == obj


def test_save_load_h5_with_wrong_model(make_data_mimo, tmp_model_data_paths):
    si_model, _, _ = make_data_mimo
    model_path, extra_data_path = tmp_model_data_paths
    si_model.save_h5(model_path, extra_data_path)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The model type in the extra data file (TensorFlowModel) does not match the current one (SurrogateModel)."
        ),
    ):
        SurrogateModel.load_h5(model_path, extra_data_path)
        os.remove(model_path)
        os.remove(extra_data_path)


# We exclude this test until we will be able to drop TensorFlow 2.10.
def _test_save_h5_extra_data_to_non_h5_model_error(make_data_mimo):
    si_model, _, _ = make_data_mimo
    model_path = f"./{str(uuid.uuid4())}-model.keras"
    with pytest.raises(
        ValueError,
        match="The extra data cannot be appended to the model file, because it is not a hdf5 file. Please specify the extra_data_path argument",
    ):
        si_model.save_h5(model_path, save_format="keras")

        os.remove(model_path)


def test_eq_wrong_type(make_data_mimo):
    si_model, _, _ = make_data_mimo
    obj_2 = MinMaxScalerSI(np.array([1.1, 10.4, 3.7]), np.array([-3.0, -50.0, +21.0]))
    assert not si_model == obj_2


# %% Test SurrogateModelComp.


def test_om_compute_array(make_data_mimo):
    # Get the data.
    si_model, input_test, output_test = make_data_mimo
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
    # Check the output.
    npt.assert_allclose(problem.get_val("y"), output_test, rtol=1e-6)


def test_om_compute_dict(make_data_mimo):
    # Get the data.
    si_model, input_test, output_test = make_data_mimo
    # Make the OpenMDAO problem.
    component = SurrogateModelComp(
        model=si_model,
        input_type=InputOutputType.SPLIT,
        output_type=InputOutputType.SPLIT,
        n_points=input_test.shape[0],
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["*"], promotes_outputs=["*"]
    )
    problem.setup()
    # Set the input.
    for i in range(si_model.n_inputs):
        problem.set_val(si_model.input_names[i], input_test[:, i])
    # Evaluate the model.
    problem.run_model()
    # Get the output.
    y = np.column_stack([problem.get_val(key) for key in si_model.output_names])
    # Check the output.
    npt.assert_allclose(y, output_test, rtol=1e-6)


def test_om_setup_wrong_input_type():
    with pytest.raises(ValueError):
        SurrogateModelComp(
            model=TensorFlowModel(None, n_inputs=2, n_outputs=1),
            input_type=-5,
            output_type=InputOutputType.SPLIT,
            n_points=10,
        )


def test_om_setup_wrong_output_type():
    with pytest.raises(ValueError):
        SurrogateModelComp(
            model=TensorFlowModel(None, n_inputs=2, n_outputs=1),
            input_type=InputOutputType.SPLIT,
            output_type=-5,
            n_points=10,
        )


def test_om_compute_partials_joined_joined(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelComp(
        model=si_model,
        input_type=InputOutputType.JOINED,
        output_type=InputOutputType.JOINED,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["x"], promotes_outputs=["y"]
    )
    problem.setup()
    # Set the input.
    problem.set_val("x", input_test)
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Jacobian computed by finite differences.
    jacobian_fd = om_partials["surrogate"][("y", "x")]["J_fd"]
    # Exact Jacobian.
    jacobian_exact = om_partials["surrogate"][("y", "x")]["J_fwd"]
    # Test.
    npt.assert_allclose(jacobian_fd, jacobian_exact, rtol=5e-1, atol=2e-1)


def test_om_compute_partials_joined_split(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelComp(
        model=si_model,
        input_type=InputOutputType.JOINED,
        output_type=InputOutputType.SPLIT,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["x"], promotes_outputs=["*"]
    )
    problem.setup()
    # Set the input.
    problem.set_val("x", input_test)
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the output channels.
    for name in si_model.output_names:
        # Jacobian computed by finite differences.
        jacobian_fd = om_partials["surrogate"][(name, "x")]["J_fd"]
        # Exact Jacobian.
        jacobian_exact = om_partials["surrogate"][(name, "x")]["J_fwd"]
        # Test.
        npt.assert_allclose(jacobian_fd, jacobian_exact, rtol=5e-1, atol=2e-1)


def test_om_compute_partials_split_joined(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelComp(
        model=si_model,
        input_type=InputOutputType.SPLIT,
        output_type=InputOutputType.JOINED,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["*"], promotes_outputs=["y"]
    )
    problem.setup()
    # Set the input.
    for i in range(si_model.n_inputs):
        problem.set_val(si_model.input_names[i], input_test[:, i])
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the input channels.
    for name in si_model.input_names:
        # Jacobian computed by finite differences.
        jacobian_fd = om_partials["surrogate"][("y", name)]["J_fd"]
        # Exact Jacobian.
        jacobian_exact = om_partials["surrogate"][("y", name)]["J_fwd"]
        # Test.
        npt.assert_allclose(jacobian_fd, jacobian_exact, rtol=5e-1, atol=2e-1)


def test_om_compute_partials_split_split(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelComp(
        model=si_model,
        input_type=InputOutputType.SPLIT,
        output_type=InputOutputType.SPLIT,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["*"], promotes_outputs=["*"]
    )
    problem.setup()
    # Set the input.
    for i in range(si_model.n_inputs):
        problem.set_val(si_model.input_names[i], input_test[:, i])
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the input channels.
    for name_out in si_model.output_names:
        for name_in in si_model.input_names:
            # Jacobian computed by finite differences.
            jacobian_fd = om_partials["surrogate"][(name_out, name_in)]["J_fd"]
            # Exact Jacobian.
            jacobian_exact = om_partials["surrogate"][(name_out, name_in)]["J_fwd"]
            # Test.
            npt.assert_allclose(jacobian_fd, jacobian_exact, rtol=5e-1, atol=2e-1)


def test_om_matrix_free_joined_joined(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    input_test = input_test[:10, :]
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelCompMatrixFree(
        model=si_model,
        input_type=InputOutputType.JOINED,
        output_type=InputOutputType.JOINED,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["x"], promotes_outputs=["y"]
    )
    problem.setup()
    # Set the input.
    problem.set_val("x", input_test)
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Jacobian computed by finite differences.
    jacobian_fd = om_partials["surrogate"][("y", "x")]["J_fd"]
    # Exact Jacobian.
    jacobian_fwd = om_partials["surrogate"][("y", "x")]["J_fwd"]
    jacobian_rev = om_partials["surrogate"][("y", "x")]["J_rev"]
    # Test.
    npt.assert_allclose(jacobian_fd, jacobian_fwd, rtol=5e-1, atol=2e-1)
    npt.assert_allclose(jacobian_fd, jacobian_rev, rtol=5e-1, atol=2e-1)


def test_om_matrix_free_joined_joined_outside_of_domain(make_data_miso):
    # Get the data.
    si_model, _, _ = make_data_miso
    input = np.column_stack(
        (
            np.linspace(-6.0, 20.0, 4, dtype=np.single),
            np.linspace(-2.0, 1.0, 4, dtype=np.single),
        )
    )
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelCompMatrixFree(
        model=si_model,
        input_type=InputOutputType.JOINED,
        output_type=InputOutputType.JOINED,
        n_points=input.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["x"], promotes_outputs=["y"]
    )
    problem.setup()
    # Set the input.
    problem.set_val("x", input)
    # Check partial derivatives.
    with pytest.warns(
        UserWarning, match="The hot point is outside of the input domain."
    ):
        problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")


def test_om_matrix_free_joined_split(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    input_test = input_test[:10, :]
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelCompMatrixFree(
        model=si_model,
        input_type=InputOutputType.JOINED,
        output_type=InputOutputType.SPLIT,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["x"], promotes_outputs=["*"]
    )
    problem.setup()
    # Set the input.
    problem.set_val("x", input_test)
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the output channels.
    for name in si_model.output_names:
        # Jacobian computed by finite differences.
        jacobian_fd = om_partials["surrogate"][(name, "x")]["J_fd"]
        # Exact Jacobian.
        jacobian_fwd = om_partials["surrogate"][(name, "x")]["J_fwd"]
        jacobian_rev = om_partials["surrogate"][(name, "x")]["J_rev"]
        # Test.
        npt.assert_allclose(jacobian_fd, jacobian_fwd, rtol=5e-1, atol=2e-1)
        npt.assert_allclose(jacobian_fd, jacobian_rev, rtol=5e-1, atol=2e-1)


def test_om_matrix_free_split_joined(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    input_test = input_test[:10, :]
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelCompMatrixFree(
        model=si_model,
        input_type=InputOutputType.SPLIT,
        output_type=InputOutputType.JOINED,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["*"], promotes_outputs=["y"]
    )
    problem.setup()
    # Set the input.
    for i in range(si_model.n_inputs):
        problem.set_val(si_model.input_names[i], input_test[:, i])
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the input channels.
    for name in si_model.input_names:
        # Jacobian computed by finite differences.
        jacobian_fd = om_partials["surrogate"][("y", name)]["J_fd"]
        # Exact Jacobian.
        jacobian_fwd = om_partials["surrogate"][("y", name)]["J_fwd"]
        jacobian_rev = om_partials["surrogate"][("y", name)]["J_rev"]
        # Test.
        npt.assert_allclose(jacobian_fd, jacobian_fwd, rtol=5e-1, atol=2e-1)
        npt.assert_allclose(jacobian_fd, jacobian_rev, rtol=5e-1, atol=2e-1)


def test_om_matrix_free_split_split(make_data_mimo):
    # Get the data.
    si_model, input_test, _ = make_data_mimo
    input_test = input_test[:10, :]
    # Make the OpenMDAO problem.
    problem = om.Problem()
    component = SurrogateModelCompMatrixFree(
        model=si_model,
        input_type=InputOutputType.SPLIT,
        output_type=InputOutputType.SPLIT,
        n_points=input_test.shape[0],
    )
    problem.model.add_subsystem(
        "surrogate", component, promotes_inputs=["*"], promotes_outputs=["*"]
    )
    problem.setup()
    # Set the input.
    for i in range(si_model.n_inputs):
        problem.set_val(si_model.input_names[i], input_test[:, i])
    # Check partial derivatives.
    om_partials = problem.check_partials(out_stream=None, step=1e-3, step_calc="abs")
    # Loop over the input channels.
    for name_out in si_model.output_names:
        for name_in in si_model.input_names:
            # Jacobian computed by finite differences.
            jacobian_fd = om_partials["surrogate"][(name_out, name_in)]["J_fd"]
            # Exact Jacobian.
            jacobian_fwd = om_partials["surrogate"][(name_out, name_in)]["J_fwd"]
            jacobian_rev = om_partials["surrogate"][(name_out, name_in)]["J_rev"]
            # Test.
            npt.assert_allclose(jacobian_fd, jacobian_fwd, rtol=5e-1, atol=2e-1)
            npt.assert_allclose(jacobian_fd, jacobian_rev, rtol=5e-1, atol=2e-1)


# %% Main.

if __name__ == "__main__":

    pytest.main([__file__])
