import numpy as np

def hstack(tup):
    return np.hstack(tup)

def vstack(inputs):

    inputs_2d = []
    for inp in inputs:
        if len(inp.shape) < 2:
            inp = inp.reshape(-1, 1)
        inputs_2d.append(inp)


    return np.vstack(inputs_2d)

def d_hstack(inputs):  # Works but maybe not optimal

    # make them all at least 2d arrays
    outputs = []
    output_size = sum(inp.size for inp in inputs)

    inputs_2d = []
    for inp in inputs:
        if len(inp.shape) < 2:
            inp = inp.reshape(-1, 1)
        inputs_2d.append(inp)

    stacked_output = hstack(inputs_2d)

    column_increment = 0
    for i, inp in enumerate(inputs):
        # inp = np.atleast_2d(inp)

        if len(inp.shape) < 2:
            inp = inp.reshape(-1, 1)

        input_size = inp.size
        jac = np.zeros((output_size, input_size))

        # iterate through the elements of the input array
        it = np.nditer(inp, flags=['multi_index'])
        for x in it:
            # print("%d %d <%s>" % (x, it.iterindex, it.multi_index))
            inp_flattened_index = it.iterindex
            output_index = list(it.multi_index)
            output_index[1] += column_increment
            output_flattened_index = np.ravel_multi_index(output_index, stacked_output.shape)
            jac[output_flattened_index, inp_flattened_index] = 1.0

        if len(inp.shape) > 1:
            column_increment += inp.shape[1]
        else:
            column_increment += 1

        jac = np.reshape(jac, (output_size,) + inp.shape)
        jac = np.squeeze(jac)
        outputs.append(jac)

    return outputs

def d_vstack(inputs):  # Works but maybe not optimal

    outputs = []
    output_size = sum(inp.size for inp in inputs)

    # TODO - make a copy of the inp arrays ??
    # make them all at least 2d arrays
    inputs_2d = []
    for inp in inputs:
        if len(inp.shape) < 2:
            inp = inp.reshape(-1, 1)
        inputs_2d.append(inp)

    stacked_output = vstack(inputs_2d)

    row_increment = 0
    for i, inp in enumerate(inputs):  # TODO should this be inputs_2d ?
        # inp = np.atleast_2d(inp)

        if len(inp.shape) < 2:
            inp = inp.reshape(-1, 1)

        input_size = inp.size
        jac = np.zeros((output_size, input_size))

        # iterate through the elements of the input array
        it = np.nditer(inp, flags=['multi_index'])  # do we need to use this flag ? f_index
        for x in it:
            # print("%d %d <%s>" % (x, it.iterindex, it.multi_index))
            inp_flattened_index = it.iterindex
            output_index = list(it.multi_index)
            output_index[0] += row_increment
            output_flattened_index = np.ravel_multi_index(output_index, stacked_output.shape)
            jac[output_flattened_index, inp_flattened_index] = 1.0

        if len(inp.shape) > 1:
            row_increment += inp.shape[0]
        else:
            row_increment += 1

        jac = np.reshape(jac, (output_size,) + inp.shape)
        jac = np.squeeze(jac)
        outputs.append(jac)

    return outputs

def d_hstack_v2(inputs):

    outputs = []

    for i, input in enumerate(inputs):
        deriv_inputs_to_hstack = []
        for j, inp_inner in enumerate(inputs):
            if i == j:
                d = np.ones(inp_inner.shape)
            else:
                d = np.zeros(inp_inner.shape)
            deriv_inputs_to_hstack.append(d)
            jac = np.hstack(deriv_inputs_to_hstack)
        outputs.append(jac)

    return outputs

# def d_hstack(inputs):
#
#     outputs = []
#     total_output_size = sum(inp.size for inp in inputs)
#     # Iterate through each input array
#     for i, inp_outer in enumerate(inputs):
#
#         # build up the deriv array with flattened arrays
#         jac = np.array([])
#         for j, inp_inner in enumerate(inputs):
#             if i == j:
#                 # d = np.ravel(np.eye(inp_inner.size))
#                 d = np.eye(inp_inner.size).flatten()
#             else:
#                 # d = np.ravel(np.zeros((inp_inner.size,inp_outer.size)))
#                 d = np.zeros((inp_inner.size, inp_outer.size)).flatten()
#             jac = np.append(jac, d)
#
#         jac = np.reshape(jac, (total_output_size,) + inp_outer.shape)
#         outputs.append(jac)
#
#     return outputs
#
def d_vstack_old(inputs):

    outputs = []
    total_output_size = sum(inp.size for inp in inputs)
    # Iterate through each input array
    for i, inp_outer in enumerate(inputs):

        # build up the deriv array with flattened arrays
        jac = np.array([])
        for j, inp_inner in enumerate(inputs):
            if i == j:
                d = np.ravel(np.eye(inp_inner.size))
            else:
                d = np.ravel(np.zeros((inp_inner.size,inp_outer.size)))
            jac = np.append(jac, d)

        jac = np.reshape(jac, (total_output_size,) + inp_outer.shape)
        outputs.append(jac)

    return outputs

def d_hstack_old(tup):
    deriv = []

    total_output_size = sum(t.size for t in tup)

    # The p_tup for each input `    should have total_output_size rows
    # They should have the number of columns according to their size
    row_counter = 0
    for i, element in enumerate(tup):
        jac = np.zeros((total_output_size, element.size))
        for j in range(element.size):
            jac[row_counter + j][j] = 1.0
        row_counter += element.size
        deriv.append(jac)

    return deriv

def d_hstack_pass_by_ref(tup, d_tup):
    # TODO Check to make sure length of tup and p_tup is the same

    total_output_size = sum(t.size for t in tup)

    # The p_tup for each should have total_output_size rows
    # They should have the number of columns according to their size
    row_counter = 0
    for i, element in enumerate(tup):
        jac = np.zeros((total_output_size, element.size))
        for j in range(element.size):
            jac[row_counter + j][j] = 1.0
        row_counter += element.size
        d_tup[i] = jac


