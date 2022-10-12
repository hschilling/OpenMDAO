"""Components used mainly for testing Newton and line searches."""
import numpy as np

import openmdao.api as om


class ImplCompTwoStates(om.ImplicitComponent):
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
        self.add_output('z', 2.0, lower=1.5, upper=2.5)
        # self.add_output('z', 2.0)

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


class ImplCompTwoStatesArrays(om.ImplicitComponent):
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
        self.add_input('x', np.zeros((3, 1)))
        self.add_output('y', np.zeros((3, 1)))
        self.add_output('z', 2.0*np.ones((3, 1)), lower=1.5,
            upper=np.array([2.6, 2.5, 2.65]).reshape((3,1)))

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
        jac[('y', 'x')] = -np.diag(np.array([1.0, 1.0, 1.0]))
        jac[('y', 'y')] = np.diag(np.array([1.0, 1.0, 1.0]))
        jac[('y', 'z')] = -np.diag(np.array([2.0, 2.0, 2.0]))

        # State equation
        jac[('z', 'z')] = (inputs['x'] + 1.0) * np.eye(3)
        jac[('z', 'x')] = outputs['z'] * np.eye(3)
