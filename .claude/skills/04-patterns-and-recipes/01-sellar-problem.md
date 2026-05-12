# Sellar Problem Recipe

## Purpose
Provide a fully annotated, canonical implementation of the Sellar problem in OpenMDAO.

## Key Concepts

- Two disciplines (SellarDis1, SellarDis2) as ExplicitComponents
- Coupling via y1/y2 feedback
- Group-level solver for convergence
- Objective and constraints for optimization

## Pattern

```python
import openmdao.api as om
import numpy as np

class SellarDis1(om.ExplicitComponent):
    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('x', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('y1', val=0.0)

    def setup_partials(self):
        self.declare_partials('y1', ['z', 'x', 'y2'], method='fd')

    def compute(self, inputs, outputs):
        z = inputs['z']
        x = inputs['x']
        y2 = inputs['y2']
        outputs['y1'] = z[0]**2 + z[1] + x - 0.2*y2

class SellarDis2(om.ExplicitComponent):
    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.0)
        self.add_output('y2', val=0.0)

    def setup_partials(self):
        self.declare_partials('y2', ['z', 'y1'], method='fd')

    def compute(self, inputs, outputs):
        z = inputs['z']
        y1 = inputs['y1']
        outputs['y2'] = y1**0.5 + z[0] + z[1]

class SellarObjective(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('obj', val=0.0)

    def setup_partials(self):
        self.declare_partials('obj', ['x', 'z', 'y1', 'y2'], method='fd')

    def compute(self, inputs, outputs):
        outputs['obj'] = (inputs['x']**2 +
                          inputs['z'][1] +
                          inputs['y1'] +
                          np.exp(-inputs['y2']))

class SellarConstraints(om.ExplicitComponent):
    def setup(self):
        self.add_input('y1', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('con1', val=0.0)
        self.add_output('con2', val=0.0)

    def setup_partials(self):
        self.declare_partials('con1', 'y1')
        self.declare_partials('con2', 'y2')

    def compute(self, inputs, outputs):
        outputs['con1'] = 3.16 - inputs['y1']
        outputs['con2'] = inputs['y2'] - 24.0

prob = om.Problem()
model = prob.model

model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])
model.add_subsystem('obj_comp', SellarObjective(), promotes=['x', 'z', 'y1', 'y2', 'obj'])
model.add_subsystem('con_comp', SellarConstraints(), promotes=['y1', 'y2', 'con1', 'con2'])

model.nonlinear_solver = om.NonlinearBlockGS()
model.linear_solver = om.DirectSolver()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=0.0, upper=10.0)
prob.model.add_design_var('z', lower=-10.0, upper=10.0)
prob.model.add_objective('obj')
prob.model.add_constraint('con1', upper=0.0)
prob.model.add_constraint('con2', upper=0.0)

prob.setup()
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_driver()

print(f"Optimal x  = {prob.get_val('x')[0]:.6f}")
print(f"Optimal z  = {prob.get_val('z')}")
print(f"Minimum obj = {prob.get_val('obj')[0]:.6f}")
print(f"con1       = {prob.get_val('con1')[0]:.6f}")
print(f"con2       = {prob.get_val('con2')[0]:.6f}")
```

## Anti-Patterns

- Not using a group-level solver for coupled disciplines.
- Not promoting variables correctly.
- Skipping partial derivatives (method='fd' is OK for first pass).

## How to Guide the User

- Use this as a reference for coupled MDO problems.
- Start with method='fd', then add analytic derivatives for speed.
