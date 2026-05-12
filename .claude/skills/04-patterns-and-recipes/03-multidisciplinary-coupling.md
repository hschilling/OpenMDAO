# Multidisciplinary Coupling Recipe

## Purpose
Show how to couple multiple disciplines/components in OpenMDAO, with solvers and group hierarchy.

## Key Concepts

- Multiple ExplicitComponents with feedback
- Group-level solver for convergence
- Promoting variables for coupling

## Pattern

```python
import openmdao.api as om
import numpy as np

class AeroComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('mach', val=0.0)
        self.add_input('altitude', val=0.0)
        self.add_output('drag', val=0.0)

    def setup_partials(self):
        self.declare_partials('drag', ['mach', 'altitude'], method='fd')

    def compute(self, inputs, outputs):
        outputs['drag'] = 0.5 * inputs['mach'] * inputs['altitude']

class ThermalComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('drag', val=0.0)
        self.add_output('temperature', val=0.0)

    def setup_partials(self):
        self.declare_partials('temperature', 'drag', method='fd')

    def compute(self, inputs, outputs):
        outputs['temperature'] = 100.0 + 0.1 * inputs['drag']

class CoupledGroup(om.Group):
    def setup(self):
        self.add_subsystem('aero', AeroComp(), promotes=['mach', 'altitude', 'drag'])
        self.add_subsystem('thermal', ThermalComp(), promotes=['drag', 'temperature'])
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.DirectSolver()

prob = om.Problem()
prob.model.add_subsystem('coupled', CoupledGroup(),
                         promotes=['mach', 'altitude', 'drag', 'temperature'])

prob.setup()
prob.set_val('mach', 0.8)
prob.set_val('altitude', 10000.0)

prob.run_model()

print(f"Drag: {prob.get_val('drag')}")
print(f"Temperature: {prob.get_val('temperature')}")
```

## Anti-Patterns

- Not using a solver for coupled feedback.
- Not promoting variables correctly.

## How to Guide the User

- Use this pattern for MDA/MDO problems.
- Add more disciplines as needed.
