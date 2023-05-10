import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class ParaboloidConstraint(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['c'] = -x + y

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        partials['c', 'x'] = -1.
        partials['c', 'y'] = 1.

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

class ParaboloidConstraintWithJax(om.ExplicitComponent):
    def __init__(self):
        super(ParaboloidConstraintWithJax, self).__init__()
        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def _compute_primal(self, x, y):
        """
        This is where the jax implementation belongs.
        """
        return -x + y

    def _compute_partials_jacfwd(self, x, y):
        # deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])
        # dx, dy = deriv_func(x, y)
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['c'] = self._compute_primal(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dx, dy = self._compute_partials_jacfwd(*inputs.values())
        partials['c', 'x'] = dx
        partials['c', 'y'] = dy

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     x, y = inputs.values()
    #     dx, dy = -1., 1.0
    #
    #     partials['c', 'x'] = dx
    #     partials['c', 'y'] = dy

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

class ParaboloidWithJax(om.ExplicitComponent):
    def __init__(self):
        super(ParaboloidWithJax, self).__init__()
        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    # def compute(self, inputs, outputs):
    #     x = inputs['x']
    #     y = inputs['y']
    #
    #     outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
    #
    def _compute_primal(self, x, y):
        """
        This is where the jax implementation belongs.
        """
        return (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def _compute_partials_jacfwd(self, x, y):
        # deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])
        # dx, dy = deriv_func(x, y)
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['f_xy'] = self._compute_primal(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dx, dy = self._compute_partials_jacfwd(*inputs.values())
        partials['f_xy', 'x'] = dx
        partials['f_xy', 'y'] = dy


prob = om.Problem()
model = prob.model

model.set_input_defaults('x', val=50.)
model.set_input_defaults('y', val=50.)

# model.add_subsystem('comp', Paraboloid(), promotes=['*'])
model.add_subsystem('comp', ParaboloidWithJax(), promotes=['*'])

# model.add_subsystem('con', ParaboloidConstraint(), promotes=['*'])
model.add_subsystem('con', ParaboloidConstraintWithJax(), promotes=['*'])

prob.set_solver_print(level=0)

prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

model.add_design_var('x', lower=-50.0, upper=50.0)
model.add_design_var('y', lower=-50.0, upper=50.0)
model.add_objective('f_xy')
model.add_constraint('c', upper=-15.0)

prob.setup()

failed = prob.run_driver()

assert failed == False, "Optimization failed, info = " + str(prob.driver.pyopt_solution.optInform)
# Minimum should be at (7.166667, -7.833334)
assert_near_equal(prob['x'], 7.16667, 1e-6)
assert_near_equal(prob['y'], -7.833334, 1e-6)
