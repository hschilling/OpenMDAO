import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

def jit_stub(f, *args, **kwargs):
    return f

# This is in here as an attempt to get more precision out of jax
jax.config.update("jax_enable_x64", True)

use_analytic = True
use_jit = True

if use_jit:
    jit = jax.jit
else:
    jit = jit_stub

# Timings:

#             No jit       With jit
#
# Analytic    0.006         0.0041
#
# Jax         0.047         0.0041


class ParaboloidConstraint(om.ExplicitComponent):
    def __init__(self):
        super(ParaboloidConstraint, self).__init__()
        self.deriv_func_jacfwd = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    @partial(jit, static_argnums=(0,))
    def _compute_primal(self, x, y):
        return -x + y

    @partial(jit, static_argnums=(0,))
    def _compute_partials_jacfwd(self, x, y):
        dx, dy = self.deriv_func_jacfwd(x, y)
        return jnp.diagonal(dx), dy

    @partial(jit, static_argnums=(0,))
    def _compute_partials_analytic(self, x, y):
        dx, dy = jnp.array([[-1.]]), jnp.array([[1.]])
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['c'] = self._compute_primal(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if use_analytic:
            dx, dy = self._compute_partials_analytic(*inputs.values())
        else:
            dx, dy = self._compute_partials_jacfwd(*inputs.values())
        partials['c', 'x'] = dx
        partials['c', 'y'] = dy

class Paraboloid(om.ExplicitComponent):
    def __init__(self, **kwargs):
        _kwargs = kwargs.copy()  # so we can pass in the options arguments
        super().__init__(**_kwargs)
        # argnums indicate want derivs wrt both inputs
        self.deriv_func_jacfwd = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def initialize(self):
        # want to test the ability to have options when using jit
        self.options.declare('dummy', types=(float,), default=1234.0)

    def setup(self):
        dummy = self.options['dummy']

        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    # static_argnumns for "self" so jit can handle the options attribute of the class
    @partial(jit, static_argnums=(0,))
    def _compute_primal(self, x, y):
        dummy = self.options['dummy']  # just to try accessing this attribute
        return (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    @partial(jit, static_argnums=(0,))
    def _compute_partials_jacfwd(self, x, y):
        dx, dy = self.deriv_func_jacfwd(x, y)
        return jnp.diagonal(dx), dy

    @partial(jit, static_argnums=(0,))
    def _compute_partials_analytic(self, x, y):
        dx, dy = jnp.array([[2. * (x - 3.0) + y]]), jnp.array([[x + 2.0 * (y + 4.0)]])
        return jnp.diagonal(dx), dy

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['f_xy'] = self._compute_primal(*inputs.values())

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if use_analytic:
            dx, dy = self._compute_partials_analytic(*inputs.values())
        else:
            dx, dy = self._compute_partials_jacfwd(*inputs.values())

        partials['f_xy', 'x'] = dx
        partials['f_xy', 'y'] = dy

    def _tree_flatten(self):
        """
        Per the jax documentation, these are the attributes
        of this class that we need to reference in the jax jitted
        methods of the class.
        There are no dynamic values or arrays, only self.options is used.
        Note that we do not change the options during the evaluation of
        these methods.
        """
        children = tuple()  # arrays / dynamic values
        aux_data = {'options': self.options}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Per the jax documentation, this method is needed by jax.jit since
        we are referencing attributes of the class (self.options) in our
        jitted methods.
        """
        return cls(*children, **aux_data)

# Extends the set of types that are considered internal nodes in pytrees
jax.tree_util.register_pytree_node(Paraboloid,
                               Paraboloid._tree_flatten,
                               Paraboloid._tree_unflatten)

prob = om.Problem()
model = prob.model

model.set_input_defaults('x', val=50.)
model.set_input_defaults('y', val=50.)

model.add_subsystem('comp', Paraboloid(dummy=5678.), promotes=['*'])

model.add_subsystem('con', ParaboloidConstraint(), promotes=['*'])

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


# Run it once to see if it gets the right answer and also to have it jitted
failed = prob.run_driver()

assert failed == False, "Optimization failed, info = " + str(prob.driver.pyopt_solution.optInform)
# Minimum should be at (7.166667, -7.833334)
assert_near_equal(prob['x'], 7.16667, 1e-6)
assert_near_equal(prob['y'], -7.833334, 1e-6)

# Check to make sure the partials are being calculated correctly
with np.printoptions(linewidth=1024):
    prob.check_partials(method='cs', compact_print=False);

# time it by re-running it over and over again.
# Don't make a new problem each time because we
# don't want to time that. Just reset the problem to the original desvars and run again
import timeit
number=1000
print(timeit.timeit('reset_problem(prob); prob.run_driver()',
                    setup="from __main__ import prob, reset_problem", number=number)/number)
