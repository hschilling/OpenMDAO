# Nonlinear Solvers

## Purpose
Teach the user the difference between available nonlinear solvers, when to use each, and how to configure them for the Sellar coupled system.

## Key Concepts

### What Nonlinear Solvers Do
A nonlinear solver converges the actual variable values in a coupled system. It is needed whenever components form a circular dependency (feedback loop). Without a solver, OpenMDAO runs each component once — feedback variables never update.

### The Two Main Nonlinear Solvers

#### NonlinearBlockGaussSeidel (NLBGS)
Iterates through components one at a time, updating outputs sequentially until convergence.

```python
model.nonlinear_solver = om.NonlinearBlockGS()
model.nonlinear_solver.options['maxiter'] = 100
model.nonlinear_solver.options['atol'] = 1e-8
model.nonlinear_solver.options['rtol'] = 1e-8
model.nonlinear_solver.options['iprint'] = 2  # Print convergence history
```

**When to use:**
- Weakly coupled systems (feedback has small effect)
- When you don't have analytic derivatives yet
- As a first attempt — simple to set up

**Limitations:**
- Slow for strongly coupled systems
- May not converge at all for tight coupling
- Convergence rate is linear

#### NewtonSolver
Solves the entire coupled system simultaneously using Newton's method.

```python
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
model.nonlinear_solver.options['maxiter'] = 20
model.nonlinear_solver.options['atol'] = 1e-8
model.nonlinear_solver.options['rtol'] = 1e-8
model.nonlinear_solver.options['iprint'] = 2
model.nonlinear_solver.options['solve_subsystems'] = True  # Helps convergence
model.linear_solver = om.DirectSolver()  # Newton always needs a linear solver
```

**When to use:**
- Strongly coupled systems
- When analytic derivatives are available
- When NLBGS fails or converges too slowly

**Limitations:**
- Requires a linear solver
- More sensitive to initial conditions
- More complex to configure

### The Sellar Problem with Solvers

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
        outputs['y1'] = z[0]**2 + z[1] + inputs['x'] - 0.2*inputs['y2']


class SellarDis2(om.ExplicitComponent):

    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.0)
        self.add_output('y2', val=0.0)

    def setup_partials(self):
        self.declare_partials('y2', ['z', 'y1'], method='fd')

    def compute(self, inputs, outputs):
        z = inputs['z']
        outputs['y2'] = inputs['y1']**0.5 + z[0] + z[1]


class Sellar(om.Group):

    def setup(self):
        self.add_subsystem('d1', SellarDis1(),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2(),
                           promotes=['z', 'y1', 'y2'])

        # Assign solver to the group containing the coupled components
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['maxiter'] = 100
        self.nonlinear_solver.options['atol'] = 1e-8
        self.linear_solver = om.DirectSolver()


prob = om.Problem()
prob.model.add_subsystem('sellar', Sellar(),
                          promotes=['x', 'z', 'y1', 'y2'])

prob.setup()
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_model()

print(f"y1 = {prob.get_val('y1')}")
print(f"y2 = {prob.get_val('y2')}")
```

### Switching to Newton
Replace the solver assignment in the Group:

```python
self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
self.nonlinear_solver.options['maxiter'] = 20
self.linear_solver = om.DirectSolver()
```

### Solver Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `maxiter` | 10 | Maximum iterations before declaring failure |
| `atol` | 1e-10 | Absolute residual tolerance |
| `rtol` | 1e-10 | Relative residual tolerance (current/initial) |
| `iprint` | 1 | Verbosity: -1=none, 0=failure only, 1=final, 2=each iter |
| `solve_subsystems` | False | Newton only: run each subsystem's own solver each iteration |

## Anti-Patterns to Watch For

### Assigning Solver to Wrong Level
**Wrong instinct:** Putting the solver on `prob.model` when the coupling is inside a subgroup.
**Why it's wrong:** The solver must be on the group that directly contains the coupled components. A solver on a parent group works but is less efficient and less targeted.
**Guide the user to:** Put the solver on the lowest-level group that contains all the coupled components.

### Using NLBGS for Strongly Coupled Systems
**Wrong instinct:** Sticking with NLBGS when it is not converging.
**Guide the user to:** If NLBGS hits maxiter without converging, try Newton. If the system is strongly coupled (y1 and y2 are very sensitive to each other), Newton is the right tool.

### No Linear Solver with Newton
**Wrong instinct:** Using NewtonSolver without setting a linear solver.
**Why it's wrong:** Newton requires solving a linear system at each iteration. Without an explicit linear solver, it will use the default which may be inadequate.
**Guide the user to:** Always pair NewtonSolver with an explicit linear solver — DirectSolver for small/medium systems.

## How to Guide the User
- Start with NLBGS — it is simpler and often sufficient
- If NLBGS fails, diagnose first: is the system strongly coupled? Are initial values reasonable?
- Transition to Newton when NLBGS is confirmed insufficient
- Always remind: solvers go on the Group, not on individual components
- Use iprint=2 to show convergence history — it helps users understand what the solver is doing
