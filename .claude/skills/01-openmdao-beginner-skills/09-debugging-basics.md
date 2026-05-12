# Debugging Basics

## Purpose
Equip the user with the essential debugging tools in OpenMDAO so they can diagnose and fix common issues on their own.

## Key Concepts

### Tool 1: The N2 Diagram
The N2 (N-squared) diagram is the single most useful debugging tool. It shows your entire model as an interactive matrix: components on the diagonal, connections as off-diagonal entries.

```python
prob.setup()
prob.run_model()

om.n2(prob)  # Opens an interactive HTML diagram in your browser
```

What to look for:
- **Diagonal blocks** = components, in execution order
- **Off-diagonal entries** = connections between components
- **Above the diagonal** = feedback connections (coupling — these need solvers)
- **Missing connections** = an input you expected to be connected is using its default value

### Tool 2: list_connections()
Shows all connections in your model:
```python
prob.setup()
prob.model.list_connections()
```

Use this to verify:
- All intended connections exist
- No unintended connections from sloppy promotes
- Source and target variable names are correct

### Tool 3: check_config()
Runs configuration checks and reports warnings:
```python
prob.setup()
prob.run_model()

# Check configuration issues
prob.check_config(checks=['all'], out_file='check_config.log')
```

Common warnings:
- Unconnected inputs (using default values)
- Missing recorder setup
- Solver configuration issues

### Tool 4: list_inputs() and list_outputs()
Inspect the actual values in the model after running:
```python
prob.run_model()

# Show all inputs and their values
prob.model.list_inputs()

# Show all outputs and their values
prob.model.list_outputs()

# Show values for a specific component
prob.model.parab.list_inputs()
prob.model.parab.list_outputs()
```

### Tool 5: check_partials()
Verify that your declared derivatives are correct by comparing against finite difference:
```python
prob.setup()
prob.run_model()

data = prob.check_partials()
```

This prints a report showing:
- Your declared derivative value
- The finite-difference approximation
- The absolute and relative error between them

If errors are large (relative error > 1e-5), your analytic derivatives have a bug.

## Common Error Messages Decoded

| Error Message | Likely Cause | Fix |
|--------------|-------------|-----|
| `KeyError: 'variable_name'` | Typo in variable name or wrong path | Check spelling, check if promotes changed the path |
| `RuntimeError: ... has no connected output` | An input is not connected to any output | Add a `connect()` or check `promotes` |
| `SetupError: ... multiple outputs ...` | Two components promote an output with the same name | Rename one output or use explicit `connect()` |
| `SolverError: ... failed to converge` | Nonlinear solver did not reach tolerance | See convergence fixes in 08-understanding-convergence |
| `ValueError: shape mismatch` | Connected variables have different shapes | Check `val=` declarations match in shape |

## Debugging Workflow
When something goes wrong, follow this sequence:

1. **Read the error message** — OpenMDAO error messages are usually descriptive
2. **Run `om.n2(prob)`** — Visualize the model, check connections
3. **Run `prob.model.list_connections()`** — Verify connections are correct
4. **Run `prob.model.list_inputs()`** — Check that inputs have expected values (not defaults)
5. **Run `prob.check_partials()`** — If optimizing, verify derivatives
6. **Simplify** — Comment out components, test pieces individually

## How to Guide the User
- If the user says "it's not working" without details, ask them to share the error message first
- If they have no error but wrong results, suggest `list_inputs()` and `list_outputs()` to inspect values
- If they suspect a connection issue, suggest `list_connections()` and the N2 diagram
- If optimization fails or gives wrong results, suggest `check_partials()`
- Always encourage the N2 diagram — it turns abstract model structure into something visual and debuggable
- If the user is overwhelmed, remind them: "Start with the N2 diagram. It shows you everything."
