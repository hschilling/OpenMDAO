# Running Your Model

## Purpose
Teach the user the execution lifecycle — setup, set values, run, get results — and the difference between run_model() and run_driver().

## Key Concepts

### The Execution Lifecycle
Every OpenMDAO session follows this pattern:

```python
import openmdao.api as om

# 1. Create the Problem
prob = om.Problem()

# 2. Build the model — add components, groups, connections
prob.model.add_subsystem('parab', Paraboloid(),
                          promotes_inputs=['x', 'y'],
                          promotes_outputs=['f_xy'])

# 3. Setup — OpenMDAO analyzes structure, allocates memory
prob.setup()

# 4. Set input values
prob.set_val('x', 3.0)
prob.set_val('y', -4.0)

# 5. Run
prob.run_model()

# 6. Get results
print(prob.get_val('f_xy'))
```

### setup() — What Happens Behind the Scenes
When you call `prob.setup()`:
- OpenMDAO validates all components, connections, and variable declarations
- It determines the execution order from the connection graph
- It allocates memory for all variables
- After `setup()`, you can set values and run — but you cannot add new components

### run_model() vs run_driver()
- **`run_model()`** — Executes the model once (or until solvers converge). Use this for testing and verification.
- **`run_driver()`** — Executes the driver, which may call `run_model()` many times. Use this for optimization, DOE, or any driver-based study.

```python
# Just run the model once to see outputs
prob.run_model()

# Run the optimizer (which calls run_model repeatedly)
prob.run_driver()
```

### Reading Results
```python
# Get a value in its declared units
print(prob.get_val('f_xy'))

# Get a value in specific units (auto-converts)
print(prob.get_val('flight.distance', units='ft'))

# Get multiple values
print(prob.get_val('x'), prob.get_val('y'), prob.get_val('f_xy'))
```

### The Complete Paraboloid Example
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
prob.setup()

prob.set_val('x', 5.0)
prob.set_val('y', -2.0)

prob.run_model()

print(f"f({prob.get_val('x')}, {prob.get_val('y')}) = {prob.get_val('f_xy')}")
```

## Anti-Patterns to Watch For

### Adding Components After setup()
**Wrong instinct:** Building the model incrementally, calling setup, adding more, calling setup again.
**Guide the user to:** Build the entire model structure first, then call `setup()` once.

### Forgetting to Call setup()
**Wrong instinct:** Setting values immediately after adding components.
**Guide the user to:** `setup()` must be called before `set_val()`, `run_model()`, or `run_driver()`.

## How to Guide the User
- Always show the complete lifecycle: create → build → setup → set values → run → get results
- If the user is just testing a component, show `run_model()`
- If the user is optimizing, show `run_driver()`
- If the user gets errors about variables not found, check: did they call `setup()`? Are variable paths correct?
