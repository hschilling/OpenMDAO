# Design of Experiments (DOE)

## Purpose
Teach the user how to use DOEDriver to run a sweep of model evaluations across a design space, and how to read back the results using a case reader.

## Key Concepts

### What Is DOE?
Design of Experiments (DOE) runs your model at multiple combinations of input values, sampling the design space. Unlike optimization (which finds one best answer), DOE maps out how outputs vary across a range of inputs.

Use DOE when:
- You want to understand model behavior across a range of inputs
- You are building a surrogate model
- You want to visualize the response surface
- You need sensitivity information before committing to optimization

### Basic DOE Setup with Paraboloid

```python
import openmdao.api as om
from openmdao.drivers.doe_generators import UniformGenerator

prob = om.Problem()
model = prob.model

model.add_subsystem('parab', Paraboloid(),
                    promotes_inputs=['x', 'y'],
                    promotes_outputs=['f_xy'])

# Use DOEDriver
prob.driver = om.DOEDriver(UniformGenerator(num_samples=25))

# Declare design variables (the inputs to sweep)
prob.model.add_design_var('x', lower=-5.0, upper=5.0)
prob.model.add_design_var('y', lower=-5.0, upper=5.0)

# Set up a case recorder to save results
recorder = om.SqliteRecorder('doe_results.sql')
prob.driver.add_recorder(recorder)

prob.setup()
prob.run_driver()
prob.record('final')
prob.cleanup()

print("DOE complete. Results saved to doe_results.sql")
```

### DOE Generators
OpenMDAO provides several sampling strategies:

```python
from openmdao.drivers.doe_generators import (
    UniformGenerator,       # Random uniform sampling
    FullFactorialGenerator, # All combinations on a grid
    LatinHypercubeGenerator # Space-filling Latin Hypercube
)

# Uniform random: 25 random samples
prob.driver = om.DOEDriver(UniformGenerator(num_samples=25))

# Full factorial: 5 levels per variable (5^n_vars total runs)
prob.driver = om.DOEDriver(FullFactorialGenerator(levels=5))

# Latin Hypercube: 25 space-filling samples
prob.driver = om.DOEDriver(LatinHypercubeGenerator(samples=25))
```

### Reading DOE Results

```python
import openmdao.api as om

# Open the recorded database
cr = om.CaseReader('doe_results.sql')

# Get all driver cases
cases = cr.list_cases('driver')
print(f"Number of cases: {len(cases)}")

# Read each case
for case_id in cases:
    case = cr.get_case(case_id)
    x = case['x']
    y = case['y']
    f = case['f_xy']
    print(f"x={x:.3f}, y={y:.3f}, f_xy={f:.3f}")
```

### CSV-Based DOE
You can also define exact sample points using a CSV file:

```python
from openmdao.drivers.doe_generators import CSVGenerator

# my_samples.csv:
# x,y
# 1.0,2.0
# 3.0,4.0
# -1.0,0.5

prob.driver = om.DOEDriver(CSVGenerator('my_samples.csv'))
```

## Anti-Patterns to Watch For

### Confusing DOE with Optimization
**Wrong instinct:** "I'll run DOE to find the optimal design."
**Why it's wrong:** DOE samples the space — it does not search for a minimum. The minimum from a DOE is only as good as your sampling density.
**Guide the user to:** Use DOE to understand the design space, then use an optimizer (ScipyOptimizeDriver) to find the true optimum.

### Too Many Samples with Full Factorial
**Wrong instinct:** Using FullFactorialGenerator with many design variables.
**Why it's wrong:** Full factorial scales exponentially. 5 levels with 5 variables = 5^5 = 3125 runs.
**Guide the user to:** Use LatinHypercubeGenerator for more than 3 design variables. It gives good space coverage with far fewer samples.

### Forgetting the Recorder
**Wrong instinct:** Running DOE without a recorder, trying to collect results manually.
**Why it's wrong:** DOE results are only accessible after the run if they were recorded. Without a recorder, the data is lost.
**Guide the user to:** Always add a SqliteRecorder before running DOE.

## How to Guide the User
- Start with UniformGenerator or FullFactorialGenerator — they are the simplest to understand
- Always show the recorder setup and reading pattern together with the DOE setup
- If the user has more than 3 design variables, recommend LatinHypercube
- Clarify the DOE vs optimization distinction upfront — it is a very common confusion
