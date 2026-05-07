# check_totals()

## Purpose
Teach the user how to verify total derivatives across the entire model using check_totals(), why this is essential before trusting optimizer results, and how to interpret the output.

## Key Concepts

### What Are Total Derivatives?
Total derivatives are the derivatives of the objective and constraints with respect to the design variables, accounting for all the coupling and connections in the model. These are what the optimizer uses to determine which direction to move the design variables.

- **Partial derivatives** — derivatives within a single component (checked with `check_partials()`)
- **Total derivatives** — derivatives through the entire connected model (checked with `check_totals()`)

Both must be correct for optimization to work properly.

### When to Use check_totals()
- After implementing analytic derivatives and verifying with check_partials()
- Before running a long optimization — verify gradients are correct first
- When optimization gives unexpected results
- Any time you change the model structure or add new components

### Basic Usage

```python
import openmdao.api as om
import numpy as np

# (Build Sellar model as in 04-linear-solvers.md)

prob.setup(force_alloc_complex=True)
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_model()

# Check total derivatives
data = prob.check_totals(
    of=['obj', 'con1', 'con2'],   # outputs (objective + constraints)
    wrt=['x', 'z'],                # inputs (design variables)
    method='cs',                   # complex step for reference
    compact_print=True
)
```

### Understanding the Output

```
-----------------------------------------
Full Model: dy/dx
-----------------------------------------
                   J_fwd          J_fd        abs err      rel err
(obj, x)       [  2.09614]   [  2.09614]   8.88e-10     4.24e-10   OK
(obj, z)       [ -3.10996    [ -3.10996    1.28e-09     4.12e-10   OK
               [ 22.02933]   [ 22.02933]
(con1, x)      [ -0.01960]   [ -0.01960]   2.34e-12     1.19e-10   OK
(con2, x)      [  0.00980]   [  0.00980]   1.78e-11     1.82e-09   OK
-----------------------------------------
```

- **J_fwd**: your model's computed total derivative
- **J_fd / J_cs**: finite difference or complex step reference
- **abs err**: absolute error between the two
- **rel err**: relative error — this is the key metric
- **OK**: relative error is below threshold (typically 1e-6)

### Interpreting Results

| Relative Error | Meaning |
|---------------|---------|
| < 1e-6 | Good — derivatives are correct |
| 1e-6 to 1e-3 | Warning — investigate further |
| > 1e-3 | Problem — derivatives have a bug |

### Using Complex Step for Accuracy
Complex step (`method='cs'`) gives a much more accurate reference than finite difference:

```python
# Requires force_alloc_complex=True in setup()
prob.setup(force_alloc_complex=True)
data = prob.check_totals(method='cs', compact_print=True)
```

Always use complex step when available — it avoids the step-size sensitivity of finite difference.

### Workflow: Derivative Verification Chain

Follow this order when verifying derivatives:

```
1. Implement analytic partials in each component
2. Run check_partials() — verify each component in isolation
3. Build the full connected model with solvers
4. Run check_totals() — verify total derivatives through the whole system
5. Run optimization — now you can trust the results
```

### Common check_totals() Failures and Fixes

#### Large errors in all total derivatives
- Likely cause: a component's partials are wrong
- Fix: run check_partials() to identify the problem component

#### Large errors in derivatives involving coupled variables
- Likely cause: linear solver is not converging tightly enough
- Fix: tighten linear solver tolerance

```python
model.linear_solver.options['atol'] = 1e-12
model.linear_solver.options['rtol'] = 1e-12
```

#### Errors only for specific design variable/output pairs
- Likely cause: the partial for that specific path is wrong
- Fix: trace the connection path and check each component's partial along it

## Anti-Patterns to Watch For

### Skipping check_totals() Before Optimization
**Wrong instinct:** "check_partials() passed, so the optimizer will work."
**Why it's wrong:** Individual component partials can be correct but total derivatives can still be wrong due to solver issues, missing partials in a connected component, or incorrect linear solver convergence.
**Guide the user to:** Always run check_totals() before a serious optimization. It takes seconds and can save hours of debugging.

### Only Checking a Subset of Design Variables
**Wrong instinct:** Only checking `wrt=['x']` when there are also `z` design variables.
**Guide the user to:** Check all design variables and all objectives/constraints together. A problem in one path may not show up if you only check part of the model.

### Trusting Finite Difference as the Reference
**Wrong instinct:** Using `method='fd'` as the reference in check_totals().
**Why it's wrong:** Finite difference is itself approximate. If the step size is wrong, the reference is wrong.
**Guide the user to:** Use `method='cs'` (complex step) with `force_alloc_complex=True` for the most reliable reference.

## How to Guide the User
- Position check_totals() as the final verification step before running optimization — make it a habit
- Show the complete verification workflow: check_partials() → check_totals() → run_driver()
- If check_totals() fails, help the user trace which component or connection is causing the error
- Remind users that check_totals() needs the model to have been run first (run_model() before check_totals())
- If the user skips check_totals() and has optimization problems, suggest going back and running it
