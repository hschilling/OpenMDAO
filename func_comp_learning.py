import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
import openmdao.func_api as omf


def func(x,y):
    c = - x + y
    return c

class SimpleConstraint(om.ExplicitComponent):
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
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['c', 'x'] = -1.
        partials['c', 'y'] = 1.



# def func(a, b, c):
#     x = 2. * a * b + 3. * c
#     return x


# f = omf.wrap(func).defaults(shape=shape).declare_partials(of='*', wrt='*', method='jax')
# p = om.Problem()
# p.model.add_subsystem('comp', om.ExplicitFuncComp(f, use_jax=True, use_jit=use_jit))


f = omf.wrap(func).declare_partials(of='*', wrt='*', method='jax')


prob = om.Problem()
model = prob.model

model.set_input_defaults('x', val=50.)
model.set_input_defaults('y', val=50.)

model.add_subsystem('comp', Paraboloid(), promotes=['*'])
# model.add_subsystem('con', om.ExplicitFuncComp(func), promotes=['*'])

model.add_subsystem('con', om.ExplicitFuncComp(f, use_jax=True), promotes=['*'])
# model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])
# model.add_subsystem('con', SimpleConstraint(), promotes=['*'])

prob.set_solver_print(level=0)

prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

model.add_design_var('x', lower=-50.0, upper=50.0)
model.add_design_var('y', lower=-50.0, upper=50.0)
model.add_objective('f_xy')
model.add_constraint('c', upper=-15.0)

prob.setup()

failed = prob.run_driver()

