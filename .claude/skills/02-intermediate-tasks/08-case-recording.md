# Case Recording

## Purpose
Teach the user how to record model and driver data during a run and how to read it back afterward for analysis and post-processing.

## Key Concepts

### What Is Case Recording?
Case recording saves variable values at specific points during model execution to a SQLite database file. You can then read the data back after the run for analysis, plotting, or debugging.

Record when:
- Running optimizations (track convergence history)
- Running DOE (save all sample results)
- Debugging (inspect variable values at each iteration)

### Setting Up a Recorder

```python
import openmdao.api as om
import numpy as np

prob = om.Problem()
model = prob.model

# ... add subsystems, driver, design vars, etc. ...

# Create a recorder
recorder = om.SqliteRecorder('my_results.sql')

# Attach to driver (records each optimizer iteration)
prob.driver.add_recorder(recorder)

# Optionally also attach to model (records each run_model call)
prob.model.add_recorder(recorder)

prob.setup()
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_driver()

# Always record the final state and cleanup
prob.record('final')
prob.cleanup()
```

### What Gets Recorded
By default, recorders capture:
- All design variables
- All objectives
- All constraints
- Iteration counter and timestamp

To record all outputs:
```python
recorder = om.SqliteRecorder('my_results.sql')
recorder.options['record_outputs'] = True
recorder.options['record_inputs'] = True
```

### Reading Recorded Data

```python
import openmdao.api as om

cr = om.CaseReader('my_results.sql')

# List all available sources
print(cr.list_sources())

# Get driver cases (optimization iterations)
driver_cases = cr.list_cases('driver')
print(f"Number of iterations: {len(driver_cases)}")

# Read a specific case
last_case = cr.get_case(driver_cases[-1])
print(f"Final x: {last_case['x']}")
print(f"Final obj: {last_case['obj']}")

# Read all cases into a loop
for case_id in driver_cases:
    case = cr.get_case(case_id)
    print(f"obj = {case['obj']:.6f}")
```

### Tracking Optimization Convergence

```python
import openmdao.api as om
import matplotlib.pyplot as plt

cr = om.CaseReader('sellar_opt.sql')
cases = cr.list_cases('driver')

obj_history = []
for case_id in cases:
    case = cr.get_case(case_id)
    obj_history.append(case['obj'][0])

plt.plot(obj_history)
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.title('Optimization Convergence')
plt.grid(True)
plt.show()
```

### Recording System-Level Data
Attach a recorder to a specific subsystem to capture internal variables:

```python
# Record data from a specific component
prob.model.d1.add_recorder(recorder)
```

## Anti-Patterns to Watch For

### Forgetting prob.cleanup()
**Wrong instinct:** Ending the script without calling `prob.cleanup()`.
**Why it's wrong:** The SQLite database may not be properly closed, leading to corrupt or incomplete data.
**Guide the user to:** Always end with `prob.record('final')` followed by `prob.cleanup()`.

### Adding Recorder After setup()
**Wrong instinct:** Adding the recorder after calling `prob.setup()`.
**Why it's wrong:** Recorders must be attached before `setup()` to be properly initialized.
**Guide the user to:** Always add recorders before calling `prob.setup()`.

### Overwriting Existing Database
**Wrong instinct:** Running the script twice without changing the database filename.
**Why it's wrong:** OpenMDAO will raise an error if the database already exists.
**Guide the user to:** Either delete the old file first or use a new filename for each run:
```python
import os
if os.path.exists('results.sql'):
    os.remove('results.sql')
```

## How to Guide the User
- Always show the complete record cycle: add_recorder → setup → run → record('final') → cleanup
- If the user is running optimization, attach the recorder to the driver
- If the user wants to inspect internal variables, attach to the model or specific subsystem
- Show the case reader pattern immediately after the recording pattern — they always go together
