# Variables and Units

## Purpose
Teach the user how OpenMDAO's variable system works — how to declare inputs and outputs correctly, use units, set defaults, and understand how data flows through the variable system.

## Key Concepts

### Declaring Inputs and Outputs
In `setup()`, you declare variables using:
- `self.add_input('name', val=default_value)` — for inputs
- `self.add_output('name', val=default_value)` — for outputs

The `val` argument sets the default value AND tells OpenMDAO the shape/type:
- `val=0.0` → scalar float
- `val=np.zeros(3)` → array of length 3
- `val=np.zeros((3, 3))` → 3x3 matrix

### Units
OpenMDAO has a built-in unit system. When you declare a variable with units, OpenMDAO automatically converts values when connected variables have different but compatible units.

```python
def setup(self):
    self.add_input('velocity', val=0.0, units='m/s')
    self.add_input('duration', val=0.0, units='s')
    self.add_output('distance', val=0.0, units='m')
```

Rules:
- Units are optional but strongly recommended for physical quantities
- Units must be compatible for connected variables (you cannot connect 'kg' to 'm')
- Inside `compute()`, values are ALWAYS in the units you declared — OpenMDAO handles conversion at the connection boundary
- Common units: 'm', 'ft', 'kg', 'lb', 's', 'min', 'deg', 'rad', 'N', 'lbf', 'm/s', 'ft/s', 'Pa', 'psi'

### Accessing Variables in compute()
```python
def compute(self, inputs, outputs):
    # Read inputs using their declared names
    v = inputs['velocity']
    t = inputs['duration']

    # Write outputs using their declared names
    outputs['distance'] = v * t
```

- `inputs` and `outputs` behave like dictionaries
- For arrays: `inputs['velocity']` returns a numpy array
- Always use the exact string name you declared in `setup()`

### Setting Values Before Running
After `setup()`, you can set input values on the problem:
```python
prob = om.Problem()
prob.model.add_subsystem('my_comp', MyComponent())
prob.setup()

prob.set_val('my_comp.velocity', 10.0, units='m/s')
prob.set_val('my_comp.duration', 5.0, units='s')

prob.run_model()

print(prob.get_val('my_comp.distance', units='m'))
```

- `set_val()` and `get_val()` accept an optional `units` argument
- OpenMDAO converts automatically if the units differ from the declared units
- Variable paths use dot notation: `'group_name.component_name.variable_name'`

## Anti-Patterns to Watch For

### Hardcoding Unit Conversions
**Wrong instinct:** "I'll convert from feet to meters myself inside compute()."
**Why it's wrong:** OpenMDAO handles unit conversions automatically at connection boundaries. Manual conversion leads to double-conversion bugs.
**Guide the user to:** Declare units on variables and let OpenMDAO convert.

### Mismatched Shapes
**Wrong instinct:** Declaring `val=0.0` but then trying to assign an array in `compute()`.
**Guide the user to:** The default value's shape must match what `compute()` will produce. If the output is an array, declare it as one.

## How to Guide the User
- If the user asks about units, emphasize that OpenMDAO handles conversion — they just need to declare correctly
- If the user is working with physical quantities and NOT using units, suggest they add them
- If the user is confused about variable paths, explain the dot notation: group.component.variable
- When the user asks about array variables, show the `val=np.zeros(n)` pattern
