# Scaling

## Purpose
Teach the user why scaling matters for optimization, how to apply scaling to design variables, objectives, and constraints, and how to diagnose scaling problems.

## Key Concepts

### Why Scaling Matters
Gradient-based optimizers work best when all variables and outputs are of similar magnitude — ideally order 1 (between 0.1 and 10). When variables differ by orders of magnitude, the optimizer:
- Takes poorly-sized steps
- Struggles to converge
- May fail entirely or find wrong answers

Scaling is not about changing the physics — it is about presenting the problem to the optimizer in a well-conditioned form.

### The Golden Rule
**All design variables, objectives, and constraints should be scaled to be approximately order 1 from the optimizer's perspective.**

### How Scaling Works in OpenMDAO
OpenMDAO applies scaling at the optimizer interface. You declare scaling using `ref` and `ref0`:

- `ref` — the value of the variable that should map to 1.0 in the optimizer's space
- `ref0` — the value that should map to 0.0 in the optimizer's space (optional)
- `scaler` — a simple multiplier alternative to ref (optimizer_value = physical_value * scaler)

```
optimizer_value = (physical_value - ref0) / (ref - ref0)
```

### Scaling Design Variables

```python
# Without scaling — optimizer sees values of order 1e6
prob.model.add_design_var('altitude', lower=0.0, upper=40000.0)  # in meters

# With scaling — optimizer sees values between 0 and 1
prob.model.add_design_var('altitude',
                           lower=0.0,
                           upper=40000.0,
                           ref=40000.0,   # 40000 m maps to 1.0
                           ref0=0.0)      # 0 m maps to 0.0
```

### Scaling Objectives

```python
# Without scaling — objective is order 1e-8 (too small for optimizer)
prob.model.add_objective('drag_coefficient')

# With scaling — scale up so optimizer sees order 1
prob.model.add_objective('drag_coefficient', ref=0.01)  # 0.01 maps to 1.0
```

### Scaling Constraints

```python
# Without scaling
prob.model.add_constraint('stress', upper=1e8)  # Pascals — very large number

# With scaling
prob.model.add_constraint('stress',
                           upper=1e8,
                           ref=1e8)  # 1e8 Pa maps to 1.0 for optimizer
```

### Sellar with Scaling

```python
prob.model.add_design_var('x', lower=0.0, upper=10.0, ref=10.0)
prob.model.add_design_var('z', lower=-10.0, upper=10.0, ref=10.0)

# Objective is already order 1 for typical Sellar values — no scaling needed
prob.model.add_objective('obj')

# Constraints are order 1 — no scaling needed
prob.model.add_constraint('con1', upper=0.0)
prob.model.add_constraint('con2', upper=0.0)
```

### Diagnosing Scaling Problems

Use `check_totals()` with scaling to see what the optimizer sees:

```python
prob.run_model()
prob.check_totals(compact_print=True)
```

Also review the driver's scaling report:
```python
prob.driver.options['invalid_desvar_behavior'] = 'warn'
```

### Signs of a Scaling Problem
- Optimizer converges in 1-2 iterations (variables not moving enough)
- Optimizer takes hundreds of iterations without progress
- Optimizer hits bounds immediately
- Optimizer reports very large or very small gradient values

## Anti-Patterns to Watch For

### Skipping Scaling Entirely
**Wrong instinct:** "My model runs, so scaling must be fine."
**Why it's wrong:** Poor scaling causes silent optimizer failures — the optimizer appears to run but finds wrong answers or fails to converge.
**Guide the user to:** Always check the order of magnitude of design variables, objectives, and constraints. If any are far from order 1, add scaling.

### Using Magic Numbers for ref
**Wrong instinct:** Setting `ref=1000` without thinking about what the variable actually represents.
**Guide the user to:** Set `ref` to the expected or nominal value of the variable. If altitude ranges from 0 to 40,000 m, set `ref=40000`.

### Scaling Inputs Instead of Design Vars
**Wrong instinct:** Dividing by a constant inside compute() to scale variables.
**Why it's wrong:** This changes the physics. Scaling should only happen at the optimizer interface.
**Guide the user to:** Always apply scaling through `add_design_var()`, `add_objective()`, and `add_constraint()` — never inside component math.

## How to Guide the User
- Introduce scaling after the user has a working optimization — do not overwhelm beginners with it upfront
- Ask: "What are the expected magnitudes of your design variables, objective, and constraints?" If any are far from order 1, scaling is needed
- Use the ref = nominal value rule of thumb — it is easy to remember and usually sufficient
- If optimization is failing mysteriously, scaling is often the first thing to check
