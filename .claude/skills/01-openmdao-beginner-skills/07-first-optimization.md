# Your First Optimization

## Purpose
Teach the user how to set up and run an optimization using the Paraboloid example, introducing drivers, design variables, objectives, and constraints.

## Key Concepts

### The Four Things You Need for Optimization
1. **A Driver** — the optimizer algorithm
2. **Design Variables** — the inputs the optimizer is allowed to change
3. **An Objective** — what the optimizer is trying to minimize (or maximize)
4. **Constraints** (optional) — limits the optimizer must respect

### The Paraboloid Optimization Example
Minimize f(x, y) = (x - 3)^2 + x*y + (y + 4)^2 - 3, subject to x + y >= 1:

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

# 1. Create Problem and add the component
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(),
                          promotes_inputs=['x', 'y'],
                          promotes_outputs=['f_xy'])

# 2. Set up the driver (optimizer)
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# 3. Declare design variables with bounds
prob.model.add_design_var('x', lower=-50.0, upper=50.0)
prob.model.add_design_var('y', lower=-50.0, upper=50.0)

# 4. Declare the objective
prob.model.add_objective('f_xy')

# 5. Declare constraints (optional)
prob.model.add_constraint('x', lower=-50.0)  # Example bound constraint

# 6. Setup and run
prob.setup()
prob.set_val('x', 3.0)  # Initial guess
prob.set_val('y', -4.0)  # Initial guess

prob.run_driver()

print(f"Optimal x = {prob.get_val('x')}")
print(f"Optimal y = {prob.get_val('y')}")
print(f"Minimum f_xy = {prob.get_val('f_xy')}")
```

### Key Details

**Design Variables:**
```python
prob.model.add_design_var('x', lower=-50.0, upper=50.0)
```
- The variable must be an input (or promoted to model level)
- `lower` and `upper` set bounds the optimizer must respect
- Initial value is set via `prob.set_val()` — this is the starting point for the optimizer

**Objectives:**
```python
prob.model.add_objective('f_xy')
```
- Must be a single scalar output
- By default, the optimizer minimizes. To maximize: `add_objective('f_xy', ref=-1.0)` or use `scaler=-1.0`

**Constraints:**
```python
prob.model.add_constraint('some_output', lower=0.0)   # some_output >= 0
prob.model.add_constraint('some_output', upper=10.0)   # some_output <= 10
prob.model.add_constraint('some_output', equals=5.0)   # some_output == 5
```

### Available Drivers
- `om.ScipyOptimizeDriver()` — wraps SciPy optimizers (SLSQP, COBYLA, etc.). Good starting point.
- `om.DOEDriver()` — Design of Experiments, not optimization
- `om.pyOptSparseDriver()` — wraps pyOptSparse for more advanced optimizers (SNOPT, IPOPT). Requires separate install.

## Anti-Patterns to Watch For

### Bad Initial Guess
**Wrong instinct:** "The optimizer will find the answer regardless of where I start."
**Guide the user to:** Initial guesses matter, especially for nonlinear problems. Start from a physically reasonable point. Try multiple starting points if unsure.

### No Bounds on Design Variables
**Wrong instinct:** Leaving bounds as default (unbounded).
**Guide the user to:** Always set physically meaningful bounds. Unbounded optimization can diverge or find non-physical solutions.

## How to Guide the User
- Walk them through the four ingredients: driver, design variables, objective, constraints
- If they don't know which optimizer to use, start with `ScipyOptimizeDriver` and `SLSQP`
- Remind them that `method='fd'` in `setup_partials()` is fine for getting started but analytic derivatives will improve optimizer performance
- If optimization fails, check: Are bounds reasonable? Is the initial guess feasible? Are constraints satisfiable?
