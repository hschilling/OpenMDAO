""" Unit tests for the system interface."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates, \
    ImplCompTwoStatesArrays
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning


class ScalingExample3(ImplicitComponent):

    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=1e2, res_ref=1e5)
        self.add_output('y2', val=6000., ref=1e3, res_ref=1e-5)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1) / y1
        residuals['y2'] = 1e-5 * (x2 - y2) / y2

class ScalingExample3NoMeta(ImplicitComponent):

    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200.)
        self.add_output('y2', val=6000.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1) / y1
        residuals['y2'] = 1e-5 * (x2 - y2) / y2


class TestSystem(unittest.TestCase):

    def test_vector_context_managers(self):
        g1 = Group()
        g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b'])
        model.add_subsystem('Sink', ExecComp('c=2*b'), promotes=['b'])

        p = Problem(model=model)
        p.set_solver_print(level=0)

        # Test pre-setup errors
        with self.assertRaises(Exception) as cm:
            inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(str(cm.exception),
                         "<class Group>: Cannot get vectors because setup has not yet been called.")

        with self.assertRaises(Exception) as cm:
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors()
        self.assertEqual(str(cm.exception),
                         "<class Group>: Cannot get vectors because setup has not yet been called.")

        p.setup()
        p.run_model()

        # Test inputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], 5.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], 5.)

        # Test inputs after setting a new value
        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        inputs['C1.a'] = -1.

        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], -1.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], -1.)

        # Test outputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(outputs['G1.G2.C1.b'], 10.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()

        # Test outputs after setting a new value
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        outputs['G1.G2.C1.b'] = 123.
        self.assertEqual(outputs['G1.G2.C1.b'], 123.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        outputs['C1.b'] = 789.
        self.assertEqual(outputs['C1.b'], 789.)

        # Test residuals
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        residuals['G1.G2.C1.b'] = 99.0
        self.assertEqual(residuals['G1.G2.C1.b'], 99.0)

        # Test linear
        d_inputs, d_outputs, d_residuals = model.get_linear_vectors()
        d_outputs['G1.G2.C1.b'] = 10.
        self.assertEqual(d_outputs['G1.G2.C1.b'], 10.)

    def test_set_checks_shape(self):
        indep = IndepVarComp()
        indep.add_output('a')
        indep.add_output('x', shape=(5, 1))

        g1 = Group()
        g1.add_subsystem('Indep', indep, promotes=['a', 'x'])

        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])
        g2.add_subsystem('C2', ExecComp('y=2*x',
                                        x=np.zeros((5, 1)),
                                        y=np.zeros((5, 1))),
                                        promotes=['x', 'y'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b', 'y'])
        model.add_subsystem('Sink', ExecComp(('c=2*b', 'z=2*y'),
                                             y=np.zeros((5, 1)),
                                             z=np.zeros((5, 1))),
                                             promotes=['b', 'y'])

        p = Problem(model=model)
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        msg = "'.*' <class Group>: Failed to set value of '.*': could not broadcast input array from shape (.*) into shape (.*)."

        num_val = -10
        arr_val = -10*np.ones((5, 1))
        bad_val = -10*np.ones((10))

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        #
        # set input
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C1.a'] = arr_val

        # assign scalar to array
        inputs['C2.x'] = num_val
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign array to array
        inputs['C2.x'] = arr_val
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C2.x'] = bad_val

        # assign list to array
        inputs['C2.x'] = arr_val.tolist()
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C2.x'] = bad_val.tolist()

        #
        # set output
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C1.b'] = arr_val

        # assign scalar to array
        outputs['C2.y'] = num_val
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign array to array
        outputs['C2.y'] = arr_val
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C2.y'] = bad_val

        # assign list to array
        outputs['C2.y'] = arr_val.tolist()
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C2.y'] = bad_val.tolist()

        #
        # set residual
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C1.b'] = arr_val

        # assign scalar to array
        residuals['C2.y'] = num_val
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign array to array
        residuals['C2.y'] = arr_val
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C2.y'] = bad_val

        # assign list to array
        residuals['C2.y'] = arr_val.tolist()
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C2.y'] = bad_val.tolist()

    def test_list_inputs_output_with_includes_excludes(self):
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()
        p.run_model()

        # Inputs with no includes or excludes
        inputs = model.list_inputs(out_stream=None)
        self.assertEqual(len(inputs), 11)

        # Inputs with includes
        inputs = model.list_inputs(includes=['*V_out*'], out_stream=None)
        self.assertEqual(len(inputs), 3)

        # Inputs with includes matching a promoted name
        inputs = model.list_inputs(includes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 2)

        # Inputs with excludes
        inputs = model.list_inputs(excludes=['*V_out*'], out_stream=None)
        self.assertEqual(len(inputs), 8)

        # Inputs with excludes matching a promoted name
        inputs = model.list_inputs(excludes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 9)

        # Inputs with includes and excludes
        inputs = model.list_inputs(includes=['*V_out*'], excludes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 1)

        # Outputs with no includes or excludes. Explicit only
        outputs = model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 5)

        # Outputs with includes. Explicit only
        outputs = model.list_outputs(includes=['*I'], implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 4)

        # Outputs with excludes. Explicit only
        outputs = model.list_outputs(excludes=['circuit*'], implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 2)

    def test_list_inputs_outputs_val_deprecation(self):
        p = Problem()
        p.model.add_subsystem('comp', ExecComp('b=2*a'), promotes=['a', 'b'])
        p.setup()
        p.run_model()

        msg = "<model> <class Group>: The 'values' argument to 'list_inputs()' " \
              "is deprecated and will be removed in 4.0. Please use 'val' instead."

        with assert_warning(OMDeprecationWarning, msg):
            inputs = p.model.list_inputs(values=False, out_stream=None)
        self.assertEqual(inputs, [('comp.a', {})])

        with assert_warning(OMDeprecationWarning, msg):
            inputs = p.model.list_inputs(values=True, out_stream=None)
        self.assertEqual(inputs, [('comp.a', {'val': 1})])

        msg = "The metadata key 'value' will be deprecated in 4.0. Please use 'val'."
        with assert_warning(OMDeprecationWarning, msg):
            self.assertEqual(inputs[0][1]['value'], 1)

        msg = "<model> <class Group>: The 'values' argument to 'list_outputs()' " \
              "is deprecated and will be removed in 4.0. Please use 'val' instead."

        with assert_warning(OMDeprecationWarning, msg):
            outputs = p.model.list_outputs(values=False, out_stream=None)
        self.assertEqual(outputs, [('comp.b', {})])

        with assert_warning(OMDeprecationWarning, msg):
            outputs = p.model.list_outputs(values=True, out_stream=None)
        self.assertEqual(outputs, [('comp.b', {'val': 2})])

        msg = "The metadata key 'value' will be deprecated in 4.0. Please use 'val'."
        with assert_warning(OMDeprecationWarning, msg):
            self.assertEqual(outputs[0][1]['value'], 2)

        meta = p.model.get_io_metadata(metadata_keys=('val',))
        with assert_warning(OMDeprecationWarning, msg):
            self.assertEqual(meta['comp.a']['value'], 1)

        with assert_warning(OMDeprecationWarning, msg):
            meta = p.model.get_io_metadata(metadata_keys=('value',))
        self.assertEqual(meta['comp.a']['val'], 1)

        with assert_warning(OMDeprecationWarning, msg):
            meta = p.model.get_io_metadata(metadata_keys=('value',))
        with assert_warning(OMDeprecationWarning, msg):
            self.assertEqual(meta['comp.a']['value'], 1)

    def test_setup_check_group(self):

        class CustomGroup(Group):

            def setup(self):
                self._custom_setup = True

            def _setup_check(self):
                if not hasattr(self, '_custom_setup'):
                    raise RuntimeError(f"{self.msginfo}: You forget to call super() in setup()")

        class BadGroup(CustomGroup):

            def setup(self):
                # should call super().setup() here
                pass

        p = Problem(model=BadGroup())

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception), '<model> <class BadGroup>: You forget to call super() in setup()')

    def test_setup_check_component(self):

        class CustomComp(ExplicitComponent):

            def setup(self):
                self._custom_setup = True

            def _setup_check(self):
                if not hasattr(self, '_custom_setup'):
                    raise RuntimeError(f"{self.msginfo}: You forget to call super() in setup()")

        class BadComp(CustomComp):

            def setup(self):
                # should call super().setup() here
                pass

        p = Problem()
        p.model.add_subsystem('comp', BadComp())

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception), "'comp' <class BadComp>: You forget to call super() in setup()")

    def test_missing_source(self):
        prob = Problem()
        root = prob.model

        root.add_subsystem('initial_comp', ExecComp(['x = 10']), promotes_outputs=['x'])

        prob.setup()

        with self.assertRaises(KeyError) as cm:
            root.get_source('f')

        self.assertEqual(cm.exception.args[0], "<model> <class Group>: source for 'f' not found.")

    def test_list_inputs_before_final_setup(self):
        class SpeedComp(ExplicitComponent):

            def setup(self):
                self.add_input('distance', val=1.0, units='km')
                self.add_input('time', val=1.0, units='h')
                self.add_output('speed', val=1.0, units='km/h')

            def compute(self, inputs, outputs):
                outputs['speed'] = inputs['distance'] / inputs['time']

        prob = Problem()
        prob.model.add_subsystem('c1', SpeedComp(), promotes=['*'])
        prob.model.add_subsystem('c2', ExecComp('f=speed',speed={'units': 'm/s'}), promotes=['*'])

        prob.setup()

        msg = ("Calling `list_inputs` before `final_setup` will only "
              "display the default values of variables and will not show the result of "
              "any `set_val` calls.")

        with assert_warning(UserWarning, msg):
            prob.model.list_inputs(units=True, prom_name=True)

    #
    # def test_set_output_solver_options_top_model(self):
    #     top = Problem()
    #     top.model.add_subsystem('px', IndepVarComp('x', 1.0))
    #     comp = top.model.add_subsystem('comp', ImplCompTwoStatesNoMeta())
    #     top.model.connect('px.x', 'comp.x')
    #
    #     top.model.nonlinear_solver = BroydenSolver()
    #     top.model.nonlinear_solver.options['maxiter'] = 25
    #     top.model.nonlinear_solver.options['diverge_limit'] = 0.5
    #     top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
    #
    #     top.model.linear_solver = DirectSolver()
    #
    #     top.model.set_output_solver_options(name='comp.z', lower=1.5, upper=2.5)
    #
    #     top.setup()
    #
    #     top.set_solver_print(level=2)
    #     # Test lower bound: should go to the lower bound and stall
    #     top['px.x'] = 2.0
    #     top['comp.y'] = 0.0
    #     top['comp.z'] = 1.6
    #     top.run_model()
    #     assert_near_equal(top['comp.z'], 1.5, 1e-8)
    #
    #     # Test upper bound: should go to the upper bound and stall
    #     top['px.x'] = 0.5
    #     top['comp.y'] = 0.0
    #     top['comp.z'] = 2.4
    #     top.run_model()
    #     assert_near_equal(top['comp.z'], 2.5, 1e-8)
    #
    #
    # def test_set_output_solver_options_sub_model(self):
    #     top = Problem()
    #     top.model.add_subsystem('px', IndepVarComp('x', 1.0))
    #     comp = top.model.add_subsystem('comp', ImplCompTwoStatesNoMeta())
    #     top.model.connect('px.x', 'comp.x')
    #
    #     top.model.nonlinear_solver = BroydenSolver()
    #     top.model.nonlinear_solver.options['maxiter'] = 25
    #     top.model.nonlinear_solver.options['diverge_limit'] = 0.5
    #     top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
    #
    #     top.model.linear_solver = DirectSolver()
    #
    #     top.model.set_output_solver_options(name='comp.z', lower=1.5, upper=2.5)
    #
    #     top.setup()
    #
    #     top.set_solver_print(level=2)
    #     # Test lower bound: should go to the lower bound and stall
    #     top['px.x'] = 2.0
    #     top['comp.y'] = 0.0
    #     top['comp.z'] = 1.6
    #     top.run_model()
    #     assert_near_equal(top['comp.z'], 1.5, 1e-8)
    #
    #     # Test upper bound: should go to the upper bound and stall
    #     top['px.x'] = 0.5
    #     top['comp.y'] = 0.0
    #     top['comp.z'] = 2.4
    #     top.run_model()
    #     assert_near_equal(top['comp.z'], 2.5, 1e-8)
    #
    #
    # def test_set_output_solver_options_res_with_meta(self):
    #
    #     prob = Problem()
    #     model = prob.model
    #
    #     model.add_subsystem('p1', IndepVarComp('x1', 1.0))
    #     model.add_subsystem('p2', IndepVarComp('x2', 1.0))
    #     comp = model.add_subsystem('comp', ScalingExample3())
    #     model.connect('p1.x1', 'comp.x1')
    #     model.connect('p2.x2', 'comp.x2')
    #
    #
    #     # comp.set_output_solver_options(name='y1', ref=1e2, res_ref=1e5)
    #     # comp.set_output_solver_options(name='y2', ref=1e3, res_ref=1e-5)
    #
    #     # self.add_output('y1', val=200., ref=1e2, res_ref=1e5)
    #     # self.add_output('y2', val=6000., ref=1e3, res_ref=1e-5)
    #
    #
    #
    #
    #     prob.setup()
    #     prob.run_model()
    #
    #     model.run_apply_nonlinear()
    #
    #     with model._scaled_context_all():
    #         val = model.comp._residuals['y1']
    #         assert_near_equal(val, -.995)
    #         val = model.comp._residuals['y2']
    #         assert_near_equal(val, (1-6000.)/6000.)
    #
    #
    #
    # def test_set_output_solver_options_res_no_meta(self):
    #
    #     prob = Problem()
    #     model = prob.model
    #
    #     model.add_subsystem('p1', IndepVarComp('x1', 1.0))
    #     model.add_subsystem('p2', IndepVarComp('x2', 1.0))
    #     comp = model.add_subsystem('comp', ScalingExample3NoMeta())
    #     model.connect('p1.x1', 'comp.x1')
    #     model.connect('p2.x2', 'comp.x2')
    #
    #
    #     # comp.set_output_solver_options(name='y1', ref=1e2, res_ref=1e5)
    #     # comp.set_output_solver_options(name='y2', ref=1e3, res_ref=1e-5)
    #
    #     model.set_output_solver_options(name='comp.y1', ref=1e2, res_ref=1e5)
    #     model.set_output_solver_options(name='comp.y2', ref=1e3, res_ref=1e-5)
    #
    #
    #     comp._has_output_scaling = True
    #     comp._has_resid_scaling = True
    #
    #     model._has_output_scaling = True
    #     model._has_resid_scaling = True
    #
    #
    #     # self.add_output('y1', val=200., ref=1e2, res_ref=1e5)
    #     # self.add_output('y2', val=6000., ref=1e3, res_ref=1e-5)
    #
    #
    #
    #
    #     prob.setup()
    #     prob.run_model()
    #
    #     model.run_apply_nonlinear()
    #
    #     with model._scaled_context_all():
    #         val = model.comp._residuals['y1']
    #         assert_near_equal(val, -.995)
    #         val = model.comp._residuals['y2']
    #         assert_near_equal(val, (1-6000.)/6000.)
    #
    #
    # def test_set_output_solver_options_qqq(self):
        # top = Problem()
        # top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        # top.model.add_subsystem('comp', ImplCompTwoStates())
        # top.model.connect('px.x', 'comp.x')
        #
        # top.model.nonlinear_solver = BroydenSolver()
        # top.model.nonlinear_solver.options['maxiter'] = 25
        # top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        # top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
        #
        # top.model.linear_solver = DirectSolver()
        #
        # top.setup()
        #
        # top.set_solver_print(level=2)
        # # Test lower bound: should go to the lower bound and stall
        # top['px.x'] = 2.0
        # top['comp.y'] = 0.0
        # top['comp.z'] = 1.6
        # top.run_model()
        # assert_near_equal(top['comp.z'], 1.5, 1e-8)
        #
        # # Test upper bound: should go to the upper bound and stall
        # top['px.x'] = 0.5
        # top['comp.y'] = 0.0
        # top['comp.z'] = 2.4
        # top.run_model()
        # assert_near_equal(top['comp.z'], 2.5, 1e-8)
        #
        #
        # #
        # top = Problem()
        # top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        # top.model.add_subsystem('comp', ImplCompTwoStatesNoMeta())
        # top.model.connect('px.x', 'comp.x')
        #
        # top.model.nonlinear_solver = BroydenSolver()
        # top.model.nonlinear_solver.options['maxiter'] = 25
        # top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        # top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
        #
        # top.model.linear_solver = DirectSolver()
        #
        # top.setup()
        #
        # top.model.set_output_solver_options(name='comp.z', lower=1.5, upper=2.5)
        #
        # top.set_solver_print(level=2)
        # # Test lower bound: should go to the lower bound and stall
        # top['px.x'] = 2.0
        # top['comp.y'] = 0.0
        # top['comp.z'] = 1.6
        # top.run_model()
        # assert_near_equal(top['comp.z'], 1.5, 1e-8)
        #
        # #
        # top = Problem()
        # top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        # comp = top.model.add_subsystem('comp', ImplCompTwoStatesNoMeta())
        # top.model.connect('px.x', 'comp.x')
        #
        # top.model.nonlinear_solver = BroydenSolver()
        # top.model.nonlinear_solver.options['maxiter'] = 25
        # top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        # top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
        #
        # top.model.linear_solver = DirectSolver()
        #
        # # comp.set_output_solver_options(name='z', lower=1.5, upper=2.5)
        # top.model.set_output_solver_options(name='comp.z', lower=1.5, upper=2.5)
        #
        # top.setup()
        #
        #
        # comp._has_bounds = True
        # top.model._has_bounds = True
        #
        #
        # top.set_solver_print(level=2)
        # # Test lower bound: should go to the lower bound and stall
        # top['px.x'] = 2.0
        # top['comp.y'] = 0.0
        # top['comp.z'] = 1.6
        # top.run_model()
        # assert_near_equal(top['comp.z'], 1.5, 1e-8)
        #
        # # Test upper bound: should go to the upper bound and stall
        # top['px.x'] = 0.5
        # top['comp.y'] = 0.0
        # top['comp.z'] = 2.4
        # top.run_model()
        # assert_near_equal(top['comp.z'], 2.5, 1e-8)
        #
        # #####
        # # Now without add_output
        # #
        # # top = Problem()
        # # top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        # # # top.model.add_subsystem('comp', ImplCompTwoStatesArraysNoLowerUpperAddOutput())
        # # top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        # # top.model.connect('px.x', 'comp.x')
        # #
        # # top.model.nonlinear_solver = BroydenSolver()
        # # top.model.nonlinear_solver.options['maxiter'] = 25
        # # top.model.nonlinear_solver.options['diverge_limit'] = 0.5
        # # top.model.nonlinear_solver.options['state_vars'] = ['comp.y', 'comp.z']
        # #
        # # top.model.linear_solver = DirectSolver()
        # #
        # # top.setup()
        # #
        # # top.model.set_output_solver_options(name='comp.z', lower=1.5,
        # #                                     upper=np.array([2.6, 2.5, 2.65]).reshape((3, 1)))
        # #
        # #
        # # top.set_solver_print(level=2)
        # # # Test lower bound: should go to the lower bound and stall
        # # top['px.x'] = 2.0
        # # top['comp.y'] = 0.0
        # # top['comp.z'] = 1.6
        # # top.run_model()
        # # assert_near_equal(top['comp.z'], 1.5, 1e-8)
        # #
        # # # Test upper bound: should go to the upper bound and stall
        # # top['px.x'] = 0.5
        # # top['comp.y'] = 0.0
        # # top['comp.z'] = 2.4
        # # top.run_model()
        # # assert_near_equal(top['comp.z'], 2.5, 1e-8)
        #
        #
        #





class ImplCompTwoStatesNoMeta(ImplicitComponent):
    """
    A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol : when x = 0.5, z = 2.666
    Sol : when x = 2.0, z = 1.333

    Coupled derivs:

    y = x + 8/(x+1)
    dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

    z = 4/(x+1)
    dz_dx = -4/(x+1)**2 = -1.7777777777777777
    """

    def setup(self):
        self.add_input('x', 0.5)
        self.add_output('y', 0.0)
        # self.add_output('z', 2.0, lower=1.5, upper=2.5)
        self.add_output('z', 2.0)

        self.maxiter = 10
        self.atol = 1.0e-12

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Don't solve; just calculate the residual.
        """

        x = inputs['x']
        y = outputs['y']
        z = outputs['z']

        residuals['y'] = y - x - 2.0*z
        residuals['z'] = x*z + z - 4.0

    def linearize(self, inputs, outputs, jac):
        """
        Analytical derivatives.
        """

        # Output equation
        jac[('y', 'x')] = -1.0
        jac[('y', 'y')] = 1.0
        jac[('y', 'z')] = -2.0

        # State equation
        jac[('z', 'z')] = inputs['x'] + 1.0
        jac[('z', 'x')] = outputs['z']





if __name__ == "__main__":
    unittest.main()
