# ExecComp

## Purpose
Teach the user how to use ExecComp as a productivity shortcut for simple mathematical expressions, avoiding the need to write a full ExplicitComponent subclass.

## Key Concepts

### What Is ExecComp?
`om.ExecComp` is a built-in OpenMDAO component that lets you define simple math expressions as strings. OpenMDAO parses the expression and automatically:
- Declares the inputs and outputs
- Computes finite difference derivatives (or analytic derivatives for common functions)

Use ExecComp when:
- The math is simple and fits in one line (or a few lines)
- You do not need custom logic, conditionals, or loops
- You want to prototype quickly

### Basic Usage

```python
import openmdao.api as om

prob = om.Problem()

# Define a component using a string expression
prob.model.add_subsystem('paraboloid',
    om.ExecComp('f_xy = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0'))

prob.setup()
prob.set_val('paraboloid.x', 5.0)
prob.set_val('paraboloid.y', -2.0)
prob.run_model()

print(prob.get_val('paraboloid.f_xy'))
```

### Multiple Expressions
Pass a list of expressions to define multiple outputs:

```python
prob.model.add_subsystem('constraints',
    om.ExecComp(['con1 = 3.16 - y1',
                 'con2 = y2 - 24.0']))
```

### Specifying Units and Default Values
Override defaults for inputs and outputs using keyword arguments:

```python
prob.model.add_subsystem('kinetic_energy',
    om.ExecComp('KE = 0.5 * m * v**2',
                m={'val': 1.0, 'units': 'kg'},
                v={'val': 0.0, 'units': 'm/s'},
                KE={'val': 0.0, 'units': 'J'}))
```

### Array Variables in ExecComp

```python
import numpy as np

prob.model.add_subsystem('vec_add',
    om.ExecComp('z = x + y',
                x=np.zeros(3),
                y=np.zeros(3),
                z=np.zeros(3)))
```

### Sellar Objective as ExecComp

```python
prob.model.add_subsystem('obj_comp',
    om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                z=np.zeros(2)),
    promotes=['x', 'z', 'y1', 'y2', 'obj'])
```

### Supported Math Functions
ExecComp supports standard Python math and numpy functions:
- `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`
- `abs`, `sum`, `dot`, `cross`
- numpy indexing: `z[0]`, `z[1]`

## Anti-Patterns to Watch For

### Overusing ExecComp for Complex Logic
**Wrong instinct:** "I'll put everything in an ExecComp expression."
**Why it's wrong:** ExecComp only supports mathematical expressions. Conditionals, loops, and complex logic require a full ExplicitComponent.
**Guide the user to:** Use ExecComp for simple math only. If the expression is getting complex or hard to read, write a proper ExplicitComponent — it will be more maintainable.

### Forgetting Units in ExecComp
**Wrong instinct:** "I'll skip units since ExecComp handles it automatically."
**Why it's wrong:** ExecComp does not infer units from expressions. Without declared units, variables are dimensionless and no unit conversion happens.
**Guide the user to:** Always declare units for physical quantities, even in ExecComp.

### Using ExecComp When Performance Matters
**Wrong instinct:** Using ExecComp for a component that is called thousands of times in an optimization.
**Why it's wrong:** ExecComp uses finite difference by default, which is slower than analytic derivatives.
**Guide the user to:** For performance-critical components, write a full ExplicitComponent with analytic derivatives.

## How to Guide the User
- Show ExecComp as a rapid prototyping tool — "get the model connected first, optimize later"
- When the user has a simple single-line computation, suggest ExecComp instead of a full component
- When expressions grow beyond 2-3 lines or include logic, suggest graduating to ExplicitComponent
- The Sellar objective and constraints are good examples of appropriate ExecComp use
