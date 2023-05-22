import inspect

import openmdao.api as om

import numpy as np
from functools import partial

from openmdao.utils.assert_utils import assert_near_equal

def jit_stub(f, **kwargs):
    """
    Provide a dummy jit decorator for use if jax is not available.

    Parameters
    ----------
    f : Callable
        The function or method to be wrapped.

    Returns
    -------
    Callable
        The decorated function.
    """
    return f

import jax
import jax.numpy as jnp

jit = jit_stub
jit = jax.jit

use_analytic = False

class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

        # argnums (Union[int, Sequence[int]]) – Optional, integer or sequence of integers.
        #     Specifies which positional argument(s) to differentiate with respect to (default 0).
        #   Here we want derivs wrt to all 3 inputs
        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1, 2])

    def setup(self):

        if self._units:
            units = 'ft'
        else:
            units = None

        if self._do_scaling:
            ref = .1
        else:
            ref = 1.

        # Global Design Variable
        self.add_input('z', val=np.zeros(2), units=units)

        # Local Design Variable
        self.add_input('x', val=0., units=units)

        # Coupling parameter
        self.add_input('y2', val=1.0, units=units)

        # Coupling output
        self.add_output('y1', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Finite difference everything
        self.declare_partials('*', '*')

    def _compute_primal(self, z, x, y2):
        return z[0]**2 + z[1] + x - 0.2*y2

    @partial(jit, static_argnums=(0,))
    def _compute_primal_with_jit(self, z, x, y2):
        return z[0]**2 + z[1] + x - 0.2*y2

    def compute(self, inputs, outputs):
        outputs['y1'] = self._compute_primal(*inputs.values())

    def _compute_partials_analytic(self, z, x, y2):
        dz = np.array([[2.0 * z[0], 1.0]])
        dx = 1.0
        dy2 = -0.2
        return dz, dx, dy2

    def _compute_partials_jacfwd(self, z, x, y2):
        dz, dx, dy2 = self.deriv_func(z, x, y2)
        return dz, dx, dy2

    @partial(jit, static_argnums=(0,))
    def _compute_partials_jacfwd_with_jit(self, z, x, y2):
        dz, dx, dy2 = self.deriv_func(z, x, y2)
        return dz, dx, dy2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if use_analytic:
            dz, dx, dy2 = self._compute_partials_analytic(*inputs.values())
        else:
            dz, dx, dy2 = self._compute_partials_jacfwd_with_jit(*inputs.values())

        # print(f"Compute partials of Dis1: {dx} {dz} {dy2}")
        partials['y1', 'z'] = dz
        partials['y1', 'x'] = dx
        partials['y1', 'y2'] = dy2

class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

        # argnums (Union[int, Sequence[int]]) – Optional, integer or sequence of integers.
        #     Specifies which positional argument(s) to differentiate with respect to (default 0).
        #   Here we want derivs wrt to both inputs
        self.deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])

    def setup(self):
        if self._units:
            units = 'inch'
        else:
            units = None

        if self._do_scaling:
            ref = .18
        else:
            ref = 1.

        # Global Design Variable
        # self.add_input('z', val=np.zeros(2), units=units)
        self.add_input('z', val=jnp.zeros(2), units=units)

        # Coupling parameter
        self.add_input('y1', val=1.0, units=units)

        # Coupling output
        self.add_output('y2', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Finite difference everything
        self.declare_partials('*', '*')


    # Used in the call to jax.jacfwd

    # Do we need this to indicate that y1 is static ?
    @partial(jit, static_argnums=(0, ))  # need to include 0 since we self is a static arg.
    # Need to include y1 because of the conditional. Could use "cond"?

    def _compute_primal(self, z, y1):

        def run_true(y1):
            return - y1

        def run_false(y1):
            return y1


        y1 = jax.lax.cond(y1.real < 0.0, run_true, run_false, y1)

        # if y1.real < 0.0:
        #     y1 *= -1

        return y1**.5 + z[0] + z[1]

    # static_argnums -
    #     An optional int or collection of ints that specify which positional arguments to treat
    #     as static (compile-time constant). Operations that only depend on static arguments will
    #     be constant-folded in Python (during tracing), and so the corresponding argument values
    #     can be any Python object.
    #
    # Static arguments should be hashable, meaning both __hash__ and __eq__ are implemented,
    # and immutable. Calling the jitted function with different values for these constants will
    # trigger recompilation. Arguments that are not arrays or containers thereof must be marked
    # as static.

    # need to explain the @partial decorator

    # The partial decorator returns a new function that has the same body as the original
    # function, but with the specified arguments bound.

    # The partial decorator can be used to create a new function that can be used with other
    # decorators, such as jit.

    # This call is equivalent to
    # jit(_compute_primal_with_jit, static_argnums=(0,2))
    # @partial(jit, static_argnums=(0,2))
    @partial(jit, static_argnums=(0,) ) # need to include 0 since we self is a static arg.
    #    Need to have this for accessing options
    def _compute_primal_with_jit(self, z, y1):

        # If y1 is not marked as being static, then we get this error
        #   Abstract tracer value encountered where concrete value is expected:
        #   Traced<ShapedArray(bool[1])>with<DynamicJaxprTrace(level=1/0)>
        # The problem arose with the `bool` function.

        # Adding 2 to the static args fixes that error but causes another one.

        # Error calling compute(), Non-hashable static arguments are not supported. An error
        # occurred during a call to '_compute_primal_with_jit' while trying to hash an object of
        # type <class 'numpy.ndarray'>, [27.8]. The error was:
        # TypeError: unhashable type: 'numpy.ndarray'


        def run_true(y1):
            return - y1

        def run_false(y1):
            return y1

        # if y1.real < 0.0:
        #     y1 *= -1

        y1 = jax.lax.cond(y1.real < 0.0, run_true, run_false, y1)

        return y1**.5 + z[0] + z[1]

    def compute(self, inputs, outputs):
        # can't have arrays as static argnums
        # outputs['y2'] = self._compute_primal_with_jit(jnp.zeros(2), jnp.ones(1))
        # inputs are ndarrays of shape (2,) and (1,)

        # can't pass in y1 as an ndarray since not hashable. Since it is just a one
        # elemenbt scalar, pass in as that
        z, y1 = inputs.values()
        y1 = y1[0]


        # outputs['y2'] = self._compute_primal_with_jit(*inputs.values())
        outputs['y2'] = self._compute_primal_with_jit(z, y1)

    def _compute_partials_analytic(self, z, y1):
        if y1.real < 0.0:
            y1 *= -1
        if y1.real < 1e-8:
            y1 = 1e-8

        dy1 = .5*y1**-.5
        dz = np.array([[1.0, 1.0]])

        return dz, dy1

    def _compute_partials_jacfwd(self, z, y1):
        dz, dy1 = self.deriv_func(z, y1)
        return dz, dy1

    # @partial(jit, static_argnums=(0,2))
    @partial(jit, static_argnums=(0,))
    def _compute_partials_jacfwd_with_jit(self, z, y1):
        dz, dy1 = self.deriv_func(z, y1)
        # dz, dy1 = jax.jacfwd(self._compute_primal, argnums=[0, 1])(z, y1)
        # dy1 = .5*y1**-.5
        # dz = np.array([[1.0, 1.0]])

        return dz, dy1

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # pass in y1, which is used in a conditional, as a scalar, which is hashable
        z, y1 = inputs.values()
        y1 = y1[0]
        if use_analytic:
            dz, dy1 = self._compute_partials_analytic(z, y1)
        else:
            dz, dy1 = self._compute_partials_jacfwd_with_jit(z, y1)

        # print(f"Compute partials of Dis2: {dz} {dy1}")

        # dz, dy1 = self._compute_partials_jacfwd_with_jit(*inputs.values())
        partials['y2', 'z'] = dz
        partials['y2', 'y1'] = dy1


class SellarDerivatives(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=None,
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=None,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        self.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        obj = self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                  x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        con1 = self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                           promotes=['con1', 'y1'])
        con2 = self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                           promotes=['con2', 'y2'])

        # manually declare partials to allow graceful fallback to FD when nested under a higher
        # level complex step approximation.
        obj.declare_partials(of='*', wrt='*', method='cs')
        con1.declare_partials(of='*', wrt='*', method='cs')
        con2.declare_partials(of='*', wrt='*', method='cs')

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        if nl:
            self.nonlinear_solver = nl() if inspect.isclass(nl) else nl

        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        if ln:
            self.linear_solver = ln() if inspect.isclass(ln) else ln

        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']


def setup_problem():
    prob = om.Problem()
    prob.model = model = SellarDerivatives()

    model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
    model.add_design_var('x', lower=0.0, upper=10.0)
    model.add_objective('obj')
    model.add_constraint('con1', upper=0.0)
    model.add_constraint('con2', upper=0.0)
    model.add_constraint('x', upper=11.0, linear=True)

    prob.set_solver_print(level=0)

    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

    prob.setup(check=False, mode='fwd')
    return prob

def reset_problem(prob):
    prob.set_val('x', 1.0)
    prob.set_val('z', np.array([5.0, 2.0]))


prob = setup_problem()
failed = prob.run_driver()


assert failed == False, "Optimization failed, info = " + str(prob.driver.result.message)
assert_near_equal(prob['z'][0], 1.9776, 1e-2)
assert_near_equal(prob['z'][1], 0.0, 1e-3)
assert_near_equal(prob['x'], 0.0, 1e-3)

with np.printoptions(linewidth=1024):
    prob.check_partials(method='cs', compact_print=False);


#
import timeit
print(timeit.timeit('reset_problem(prob); prob.run_driver()', setup="from __main__ import prob, reset_problem", number=50))

# prob.run_driver()




