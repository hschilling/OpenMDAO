import inspect

import openmdao.api as om

import numpy as np

class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

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
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2

        self.execution_count += 1

class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

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
        self.add_input('z', val=np.zeros(2), units=units)

        # Coupling parameter
        self.add_input('y1', val=1.0, units=units)

        # Coupling output
        self.add_output('y2', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Finite difference everything
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1**.5 + z1 + z2

        self.execution_count += 1



class SellarDis1withDerivatives(SellarDis1):
    """
    Component containing Discipline 1 -- derivatives version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        partials['y1', 'y2'] = -0.2
        partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
        partials['y1', 'x'] = 1.0

class SellarDis2withDerivatives(SellarDis2):
    """
    Component containing Discipline 2 -- derivatives version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, J):
        """
        Jacobian for Sellar discipline 2.
        """
        y1 = inputs['y1']
        if y1.real < 0.0:
            y1 *= -1
        if y1.real < 1e-8:
            y1 = 1e-8

        J['y2', 'y1'] = .5*y1**-.5
        J['y2', 'z'] = np.array([[1.0, 1.0]])



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
        self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

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
        con1.declare_partials(of='*', wrt='*', method='jax')
        con2.declare_partials(of='*', wrt='*', method='jax')

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


prob.run_driver()
#
# import timeit
# print(timeit.timeit('prob.run_driver()', setup="from __main__ import prob", number=100))

# prob.run_driver()




