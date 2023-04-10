import unittest

import numpy
import numpy as np

import openmdao.api as om
import openmdao.func as omfunc

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

class SampleVectorHStackAndSumComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=(3,))
        self.add_input('x2', shape=(3,))
        self.add_input('z', shape=1)
        self.add_output('y', shape=1)
        self.declare_partials('*', '*')
        # Declare these once here so we don't reallocate them with each call to compute_partials
        # self._d_out = [np.zeros(3), np.zeros(2)]

    def compute(self, inputs, outputs):
        outputs['y'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2']))) * inputs['z']

    def compute_partials(self, inputs, J):
        p_output_hstack = omfunc.d_hstack((inputs['x1'], inputs['x2']) )
        stack_and_sum = omfunc.d_sum(omfunc.hstack((inputs['x1'], inputs['x2'])))
        p_x1, p_x2 = p_output_hstack
        # using the chain rule for composite functions (https://xaktly.com/ChainRule.html)
        J['y', 'x1'] = np.matmul(stack_and_sum, p_x1)  * inputs['z']
        J['y', 'x2'] = np.matmul(stack_and_sum, p_x2)  * inputs['z']
        J['y', 'z'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

class SampleMatrixHStackAndSumComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=(3,2,))
        self.add_input('x2', shape=(3,2,))
        self.add_input('z', shape=1)
        self.add_output('y', shape=1)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['y'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2']))) * inputs['z']

    def compute_partials(self, inputs, J):
        p_output_hstack = omfunc.d_hstack((inputs['x1'], inputs['x2']) )
        stack_and_sum = omfunc.d_sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

        stack_and_sum = stack_and_sum.reshape(stack_and_sum.shape[0],-1)
        p_x1, p_x2 = p_output_hstack

        p_x1 = p_x1.reshape(p_x1.shape[0], -1)
        p_x2 = p_x2.reshape(p_x2.shape[0], -1)

        # using the chain rule for composite functions (https://xaktly.com/ChainRule.html)
        J['y', 'x1'] = np.matmul(stack_and_sum, p_x1)  * inputs['z']
        J['y', 'x2'] = np.matmul(stack_and_sum, p_x2)  * inputs['z']
        J['y', 'z'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

class SampleMatrix3HStackAndSumComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=(3,2,))
        self.add_input('x2', shape=(3,2,))
        self.add_input('x3', shape=(3,2,))
        self.add_input('z', shape=1)
        self.add_output('y', shape=1)
        self.declare_partials('*', '*')
        # Declare these once here so we don't reallocate them with each call to compute_partials
        # self._d_out = [np.zeros(3), np.zeros(2)]

    def compute(self, inputs, outputs):
        outputs['y'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'], inputs['x3']))) * inputs['z']

    def compute_partials(self, inputs, J):
        p_output_hstack = omfunc.d_hstack((inputs['x1'], inputs['x2'], inputs['x3']) )
        stack_and_sum = omfunc.d_sum(omfunc.hstack((inputs['x1'], inputs['x2'], inputs['x3'])))

        # TODO a hack to see if we can make this work with 2d inputs
        stack_and_sum = stack_and_sum.reshape(stack_and_sum.shape[0],-1)
        # stack_and_sum = numpy.squeeze(stack_and_sum)


        p_x1, p_x2, p_x3 = p_output_hstack

        p_x1 = p_x1.reshape(p_x1.shape[0], -1)
        p_x2 = p_x2.reshape(p_x2.shape[0], -1)
        p_x3 = p_x3.reshape(p_x3.shape[0], -1)
        # using the chain rule for composite functions (https://xaktly.com/ChainRule.html)
        J['y', 'x1'] = np.matmul(stack_and_sum, p_x1)  * inputs['z']
        J['y', 'x2'] = np.matmul(stack_and_sum, p_x2)  * inputs['z']
        J['y', 'x3'] = np.matmul(stack_and_sum, p_x3)  * inputs['z']
        J['y', 'z'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'], inputs['x3'])))

class Sample3DMatrixHStackAndSumComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=(3,2,4))
        self.add_input('x2', shape=(3,2,4))
        self.add_input('z', shape=1)
        self.add_output('y', shape=1)
        self.declare_partials('*', '*')
        # Declare these once here so we don't reallocate them with each call to compute_partials
        # self._d_out = [np.zeros(3), np.zeros(2)]

    def compute(self, inputs, outputs):
        outputs['y'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2']))) * inputs['z']

    def compute_partials(self, inputs, J):
        p_output_hstack = omfunc.d_hstack((inputs['x1'], inputs['x2']) )
        stack_and_sum = omfunc.d_sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

        # TODO a hack to see if we can make this work with 2d inputs
        stack_and_sum = stack_and_sum.reshape(stack_and_sum.shape[0],-1)
        # stack_and_sum = numpy.squeeze(stack_and_sum)


        p_x1, p_x2 = p_output_hstack

        p_x1 = p_x1.reshape(p_x1.shape[0], -1)
        p_x2 = p_x2.reshape(p_x2.shape[0], -1)
        # using the chain rule for composite functions (https://xaktly.com/ChainRule.html)
        J['y', 'x1'] = np.matmul(stack_and_sum, p_x1)  * inputs['z']
        J['y', 'x2'] = np.matmul(stack_and_sum, p_x2)  * inputs['z']
        J['y', 'z'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

class SampleVectorVStackAndSumComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=(3,))
        self.add_input('x2', shape=(3,))
        self.add_input('z', shape=1)
        self.add_output('y', shape=1)
        self.declare_partials('*', '*')
        # Declare these once here so we don't reallocate them with each call to compute_partials
        # self._d_out = [np.zeros(3), np.zeros(2)]

    def compute(self, inputs, outputs):
        outputs['y'] = omfunc.sum(omfunc.vstack((inputs['x1'], inputs['x2']))) * inputs['z']

    def compute_partials(self, inputs, J):
        p_output_vstack = omfunc.d_vstack((inputs['x1'], inputs['x2']) )

        # rename stack_and_sum to indicate that it is a derivative p_stack_and_sum ?
        stack_and_sum = omfunc.d_sum(omfunc.vstack((inputs['x1'], inputs['x2']))) # TODO why does an input of shape (6,1) result in derivative output shape of (1,6,1) ?

        stack_and_sum = stack_and_sum.reshape(1, omfunc.vstack((inputs['x1'], inputs['x2'])).size)


        p_x1, p_x2 = p_output_vstack
        # using the chain rule for composite functions (https://xaktly.com/ChainRule.html)
        J['y', 'x1'] = np.matmul(stack_and_sum, p_x1)  * inputs['z']
        J['y', 'x2'] = np.matmul(stack_and_sum, p_x2)  * inputs['z']
        J['y', 'z'] = omfunc.sum(omfunc.hstack((inputs['x1'], inputs['x2'])))

class TestStackCompVector(unittest.TestCase):

    def test_vector_hstack(self):

        # make a comp that sums up the values from two scalars muxed together using hstack
        # and compute the partials
        p = om.Problem()

        stack_comp = p.model.add_subsystem(name='stack_comp',
                                           subsys=SampleVectorHStackAndSumComp(), promotes_outputs=['*'])
        p.setup(force_alloc_complex=True)

        z_value = 12.34
        p['stack_comp.x1'] = 2.3 * np.ones(3)
        # p['stack_comp.x1'] = np.random.random(2)
        p['stack_comp.x2'] = 7.2 * np.ones(3)
        p['stack_comp.z'] = z_value

        p.run_model()

        # Check values
        stack_and_sum = np.sum(omfunc.hstack((p['stack_comp.x1'], p['stack_comp.x2'])))
        assert_near_equal(p['y'], stack_and_sum * p['stack_comp.z'] )

        # check partials
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_matrix_hstack(self):
        # make a comp that sums up the values from two matrices muxed together using hstack
        # and compute the partials
        p = om.Problem()

        p.model.add_subsystem(name='stack_comp', subsys=SampleMatrixHStackAndSumComp(),
                                           promotes_outputs=['*'])
        p.setup(force_alloc_complex=True)

        z_value = 12.34
        p['stack_comp.x1'] = 2.3 * np.ones((3,2))
        p['stack_comp.x2'] = 7.2 * np.ones((3,2))
        p['stack_comp.z'] = z_value

        p.run_model()

        # Check values
        stack_and_sum = np.sum(omfunc.hstack((p['stack_comp.x1'], p['stack_comp.x2'])))
        assert_near_equal(p['y'], stack_and_sum * p['stack_comp.z'])

        # check partials
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_matrix_3_hstack(self):
        # make a comp that sums up the values from three matrices muxed together using hstack
        # and compute the partials
        p = om.Problem()

        stack_comp = p.model.add_subsystem(name='stack_comp',
                                           subsys=SampleMatrix3HStackAndSumComp(),
                                           promotes_outputs=['*'])
        p.setup(force_alloc_complex=True)

        z_value = 12.34
        p['stack_comp.x1'] = 2.3 * np.ones((3,2))
        p['stack_comp.x2'] = 7.2 * np.ones((3,2))
        p['stack_comp.x3'] = 9.4 * np.ones((3,2))
        p['stack_comp.z'] = z_value

        p.run_model()

        # Check values
        stack_and_sum = np.sum(omfunc.hstack((p['stack_comp.x1'], p['stack_comp.x2'], p['stack_comp.x3'])))
        assert_near_equal(p['y'], stack_and_sum * p['stack_comp.z'])

        # check partials
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_3d_matrix_hstack(self):
        # make a comp that sums up the values from two matrices muxed together using hstack
        # and compute the partials
        p = om.Problem()

        p.model.add_subsystem(name='stack_comp',
                                           subsys=Sample3DMatrixHStackAndSumComp(),
                                           promotes_outputs=['*'])
        p.setup(force_alloc_complex=True)

        z_value = 12.34
        p['stack_comp.x1'] = 2.3 * np.ones((3, 2,4))
        p['stack_comp.x2'] = 7.2 * np.ones((3, 2,4))
        p['stack_comp.z'] = z_value

        p.run_model()

        # Check values
        stack_and_sum = np.sum(omfunc.hstack((p['stack_comp.x1'], p['stack_comp.x2'])))
        assert_near_equal(p['y'], stack_and_sum * p['stack_comp.z'])

        # check partials
        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

        # check derivs

        # check derivs
        # totals = p.compute_totals(['y'], ['stack_comp.x1', 'stack_comp.x2', 'stack_comp.z'])
        # jacobian = {}
        # jacobian['y', 'stack_comp.x1'] = [[z_value,z_value]]
        # jacobian['y', 'stack_comp.x2'] = [[z_value, z_value, z_value]]
        # jacobian['y', 'stack_comp.z'] = [[stack_and_sum]]
        # assert_near_equal(totals, jacobian)


    def test_vector_vstack(self):

        # make a comp that sums up the values from two scalars muxed together using hstack
        # and compute the partials
        p = om.Problem()

        stack_comp = p.model.add_subsystem(name='stack_comp', subsys=SampleVectorVStackAndSumComp(), promotes_outputs=['*'])
        p.setup(force_alloc_complex=True)

        z_value = 12.34
        p['stack_comp.x1'] = 2.3 * np.ones(3).reshape((3,1))
        # p['stack_comp.x1'] = np.random.random(2)
        p['stack_comp.x2'] = 7.2 * np.ones(3).reshape((3,1))
        p['stack_comp.z'] = z_value

        p.run_model()

        # Check values
        stack_and_sum = np.sum(omfunc.vstack((p['stack_comp.x1'], p['stack_comp.x2'])))
        assert_near_equal(p['y'], stack_and_sum * p['stack_comp.z'] )

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

        # check derivs
        # totals = p.compute_totals(['y'], ['stack_comp.x1', 'stack_comp.x2', 'stack_comp.z'])
        # jacobian = {}
        # jacobian['y', 'stack_comp.x1'] = [[z_value,z_value]]
        # jacobian['y', 'stack_comp.x2'] = [[z_value, z_value, z_value]]
        # jacobian['y', 'stack_comp.z'] = [[stack_and_sum]]
        # assert_near_equal(totals, jacobian)



class SampleScalarStackComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x1', shape=1)
        self.add_input('x2', shape=1)
        self.add_output('output', shape=1)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        x_stacked = omfunc.hstack((inputs['x1'], inputs['x2']))

        # outputs['sum'] = np.sum(x_stacked)
        outputs['output'] = 9.8 * x_stacked[0] ** 2 + 2.6 * x_stacked[1]

    def compute_partials(self, inputs, J):
        J['output', 'x1'] = 9.8 * (2 * inputs['x1'])
        J['output', 'x2'] = 2.6 * inputs['x2']


class TestStackCompScalar(unittest.TestCase):

    def test_scalar_hstack(self):

        # make a comp that sums up the values from two scalars muxed together using hstack
        # and compute the partials
        p = om.Problem()

        stack_comp = p.model.add_subsystem(name='stack_comp', subsys=SampleScalarStackComp(), promotes_outputs=['*'])
        p.setup()

        p['stack_comp.x1'] = 2.3
        p['stack_comp.x2'] = 7.2

        p.run_model()

        totals = p.compute_totals(['output'], ['stack_comp.x1', 'stack_comp.x2'])

        assert_near_equal(p['output'], 9.8 * p['stack_comp.x1'] ** 2 + 2.6 * p['stack_comp.x2'] )

        jacobian = {}
        jacobian['output', 'stack_comp.x1'] = [[45.08]]
        jacobian['output', 'stack_comp.x2'] = [[18.72]]

        assert_near_equal(totals, jacobian)



