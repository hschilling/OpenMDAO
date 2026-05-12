# Paraboloid Optimization Recipe

## Purpose
Show a complete, annotated recipe for optimizing the paraboloid function in OpenMDAO.

## Key Concepts

- ExplicitComponent for paraboloid
- Optimization with ScipyOptimizeDriver
- Design variables, objective, constraints

## Pattern

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        self.declare_partials('f_xy', ['x', 'y'], method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(),
                         promotes_inputs=['x', 'y'],
                         promotes_outputs=['f_xy'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=-50.0, upper=50.0)
prob.model.add_design_var('y', lower=-50.0, upper=50.0)
prob.model.add_objective('f_xy')

prob.setup()
prob.set_val('x', 3.0)
prob.set_val('y', -4.0)

prob.run_driver()

print(f"Optimal x = {prob.get_val('x')}")
print(f"Optimal y = {prob.get_val('y')}")
print(f"Minimum f_xy = {prob.get_val('f_xy')}")
```

## Anti-Patterns

- Skipping setup_partials (method='fd' is OK for first pass).
- Not setting bounds on design variables.

## How to Guide the User

- Use this as a template for single-discipline optimization.
- Add constraints as needed for your problem.
