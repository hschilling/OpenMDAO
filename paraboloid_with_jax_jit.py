import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


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


# This is in here as an attempt to get more precision out of jax
jax.config.update("jax_enable_x64", True)


class ParaboloidConstraintWithJax(om.ExplicitComponent):
    def __init__(self):
        super(ParaboloidConstraintWithJax, self).__init__()
        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def initialize(self):
        self.options.declare('vec_size', types=(int,))

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal_with_jit(self, x, y):
        return -x + y

    def _compute_primal(self, x, y):
        return -x + y

    @partial(jax.jit, static_argnums=(0,))
    def _compute_partials_jacfwd_with_jit(self, x, y):
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def _compute_partials_jacfwd(self, x, y):
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def _compute_partials_analytic(self, x, y):
        dx, dy = jnp.array([[-1.]]), jnp.array([[1.]])
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['c'] = self._compute_primal_with_jit(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dx, dy = self._compute_partials_jacfwd_with_jit(*inputs.values())
        partials['c', 'x'] = dx
        partials['c', 'y'] = dy

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     x, y = inputs.values()
    #     dx, dy = -1., 1.0
    #
    #     partials['c', 'x'] = dx
    #     partials['c', 'y'] = dy

# class Paraboloid(om.ExplicitComponent):
#
#     def initialize(self):
#         self.options.declare('dummy', types=(float,), default=0.0)
#
#     def setup(self):
#         dummy = self.options['dummy']
#
#         self.add_input('x', val=0.0)
#         self.add_input('y', val=0.0)
#
#         self.add_output('f_xy', val=0.0)
#
#     def setup_partials(self):
#         # Finite difference all partials.
#         self.declare_partials('*', '*', method='fd')
#
#
#     def compute(self, inputs, outputs):
#         x = inputs['x']
#         y = inputs['y']
#
#         outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

class ParaboloidWithJax(om.ExplicitComponent):
    def __init__(self, **kwargs):
        _kwargs = kwargs.copy()

        # super(ParaboloidWithJax, self).__init__()
        super().__init__(**_kwargs)

        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    # def __init__(self, **kwargs):
    #
    #     _kwargs = kwargs.copy()
    #
    #     self.state_options = {}
    #     self._objectives = {}
    #
    #     super().__init__(**_kwargs)


    def initialize(self):
        self.options.declare('dummy', types=(float,), default=1234.0)

    def setup(self):
        dummy = self.options['dummy']

        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('*', '*')

    # def compute(self, inputs, outputs):
    #     x = inputs['x']
    #     y = inputs['y']
    #
    #     outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
    #
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal_with_jit(self, x, y):
        dummy = self.options['dummy']
        return (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def _compute_primal(self, x, y):
        dummy = self.options['dummy']
        return (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    @partial(jax.jit, static_argnums=(0,))
    def _compute_partials_jacfwd_with_jit(self, x, y):
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def _compute_partials_jacfwd(self, x, y):
        dx, dy = self.deriv_func(x, y)
        return jnp.diagonal(dx), dy

    def _compute_partials_analytic(self, x, y):
        dx, dy = jnp.array([[2. * (x - 3.0) + y]]), jnp.array([[x + 2.0 * (y + 4.0)]])
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['f_xy'] = self._compute_primal_with_jit(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dummy = self.options['dummy']

        dx, dy = self._compute_partials_jacfwd_with_jit(*inputs.values())
        partials['f_xy', 'x'] = dx
        partials['f_xy', 'y'] = dy

    def _tree_flatten(self):
        children = tuple()  # arrays / dynamic values
        aux_data = {'options': self.options}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

from jax import tree_util
tree_util.register_pytree_node(ParaboloidWithJax,
                               ParaboloidWithJax._tree_flatten,
                               ParaboloidWithJax._tree_unflatten)

prob = om.Problem()
model = prob.model

model.set_input_defaults('x', val=50.)
model.set_input_defaults('y', val=50.)

# model.add_subsystem('comp', Paraboloid(), promotes=['*'])
model.add_subsystem('comp', ParaboloidWithJax(dummy=5678.), promotes=['*'])

# model.add_subsystem('con', ParaboloidConstraint(), promotes=['*'])
model.add_subsystem('con', ParaboloidConstraintWithJax(), promotes=['*'])

prob.set_solver_print(level=0)

prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

model.add_design_var('x', lower=-50.0, upper=50.0)
model.add_design_var('y', lower=-50.0, upper=50.0)
model.add_objective('f_xy')
model.add_constraint('c', upper=-15.0)

prob.setup(force_alloc_complex=True)

def reset_problem(prob):
    prob.set_val('x', 50.0)
    prob.set_val('y', 50.0)


failed = prob.run_driver()

assert failed == False, "Optimization failed, info = " + str(prob.driver.pyopt_solution.optInform)
# Minimum should be at (7.166667, -7.833334)
assert_near_equal(prob['x'], 7.16667, 1e-6)
assert_near_equal(prob['y'], -7.833334, 1e-6)


with np.printoptions(linewidth=1024):
    prob.check_partials(method='cs', compact_print=False);

import timeit
print(timeit.timeit('reset_problem(prob); prob.run_driver()', setup="from __main__ import prob, reset_problem", number=100))
#
#
