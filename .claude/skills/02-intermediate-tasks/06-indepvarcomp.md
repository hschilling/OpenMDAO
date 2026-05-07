# IndepVarComp

## Purpose
Teach the user what IndepVarComp is, when it is still needed in modern OpenMDAO, and when it is no longer necessary.

## Key Concepts

### What Is IndepVarComp?
`IndepVarComp` is a special component that declares independent variables — inputs to the model that are not connected to any other component's output. They are the "free" variables that drive the model.

### Modern OpenMDAO: Often Not Needed
In OpenMDAO 3.x and later, you do NOT need IndepVarComp for design variables used in optimization. OpenMDAO automatically promotes unconnected inputs and allows them to be set directly via `prob.set_val()`.

```python
# Modern approach — no IndepVarComp needed
prob = om.Problem()
model = prob.model

model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

prob.setup()

# Set values directly — no IndepVarComp required
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_model()
```

### When IndepVarComp IS Still Needed

#### 1. When you want to declare units for an independent variable at the source
```python
ivc = om.IndepVarComp()
ivc.add_output('x', val=1.0, units='m')
ivc.add_output('z', val=np.zeros(2), units='m')
model.add_subsystem('ivc', ivc, promotes=['x', 'z'])
```

#### 2. When building a component or group meant to be reused
If your Group will be embedded in a larger model and needs to declare its own independent inputs explicitly, IndepVarComp makes the interface clear.

#### 3. Legacy code
Much existing OpenMDAO code uses IndepVarComp. You will see it frequently in tutorials and examples. It still works — it is just no longer required for simple cases.

### IndepVarComp Pattern (when used)

```python
import openmdao.api as om
import numpy as np

prob = om.Problem()
model = prob.model

# Declare independent variables
ivc = om.IndepVarComp()
ivc.add_output('x', val=1.0)
ivc.add_output('z', val=np.array([5.0, 2.0]))

model.add_subsystem('ivc', ivc, promotes=['x', 'z'])
model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

prob.setup()
prob.run_model()
```

### Summary: Modern vs Legacy

| Scenario | Modern Approach | Legacy Approach |
|----------|----------------|-----------------|
| Simple independent variable | `prob.set_val('x', 1.0)` | `IndepVarComp` + `add_output` |
| Design variable for optimization | `add_design_var('x')` directly | `IndepVarComp` + `add_design_var` |
| Units on independent variable | `set_val('x', 1.0, units='m')` | `IndepVarComp` with `units=` |

## Anti-Patterns to Watch For

### Adding IndepVarComp Unnecessarily
**Wrong instinct:** "I always need an IndepVarComp for my design variables."
**Why it's wrong:** Modern OpenMDAO handles unconnected inputs automatically. Adding IndepVarComp adds boilerplate without benefit in simple cases.
**Guide the user to:** Start without IndepVarComp. Add it only if you need explicit unit declarations at the source or are building a reusable component library.

### Connecting IndepVarComp Outputs to the Wrong Level
**Wrong instinct:** Adding IndepVarComp inside a subgroup but promoting variables to the wrong level.
**Guide the user to:** IndepVarComp is typically added at the top level of the model and promoted to the model level.

## How to Guide the User
- If the user is writing new code: show the modern approach (no IndepVarComp) first
- If the user is reading existing code with IndepVarComp: explain what it does and that it is still valid
- If the user asks "do I need IndepVarComp?": almost certainly not for simple optimization problems in modern OpenMDAO
