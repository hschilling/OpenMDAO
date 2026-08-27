from time import perf_counter
import numpy as np

import openmdao.api as om


    # Generate input data: 1001 evenly spaced points across each dimension
    # np.column_stack combines two 1D arrays into a 2D array with 2 columns
    # Each row represents a point (x1, x2) in the 2D domain


def kriging_surrogate_test(indices_training, indices_validation, testing_inputs, testing_outputs, inputs, outputs, domain):
    input_numvars = inputs.shape[1]
    output_numvars = outputs.shape[1]

    mm_comp = om.MetaModelUnStructuredComp(default_surrogate=om.KrigingSurrogate())
    mm_comp.add_input('x', np.zeros(input_numvars))
    mm_comp.add_output('y', np.zeros(output_numvars))

    # add it to a Problem
    prob = om.Problem()
    prob.model.add_subsystem('mm_comp', mm_comp)
    prob.setup()

    # provide training data
    mm_comp.options['train_x'] = inputs[indices_training, :]
    mm_comp.options['train_y'] = outputs[indices_training, :]


    start_time = perf_counter()

    # train the surrogate
    prob.run_model()

    end_time = perf_counter()

    print(f"fit time = {end_time-start_time}")


    # Loop through all the test inputs and compare to outputs
    errors = 0
    for input, output_actual in zip(testing_inputs, testing_outputs):
        prob.set_val('mm_comp.x', input)
        prob.run_model()
        output_predicted = prob.get_val('mm_comp.y')
        absolute_difference = np.absolute(output_actual - output_predicted)
        # keep track of the average error
        errors += np.sum(absolute_difference)

    return mm_comp.__class__.__name__, errors
