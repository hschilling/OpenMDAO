# Understanding Convergence

## Purpose
Demystify convergence for beginners — what it means, when it matters, how to read solver output, and common fixes when things don't converge.

## Key Concepts

### When Does Convergence Matter?
Convergence matters when your model has **coupling** — circular dependencies between components.

**Uncoupled model (no convergence needed):**
```
A -> B -> C (data flows one way)
```
OpenMDAO runs A, then B, then C. Done.

**Coupled model (convergence needed):**
```
A -> B -> A (circular dependency)
```
A's output feeds B, but B's output feeds back to A. OpenMDAO must iterate until values stabilize (converge).

### What Is Convergence?
Convergence means the system has reached a self-consistent state — running the components again would not change the values. Mathematically, the **residuals** (the difference between what a component expects and what it gets) are below a specified tolerance.

### Solvers: The Tools for Convergence
Solvers are assigned to **Groups** (not components). The group's solver handles convergence for all coupled components within that group.

**Nonlinear Solvers (converge values):**
```python
# Newton solver — fast, needs derivatives
group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

# Nonlinear Block Gauss-Seidel — simple, no derivatives needed
group.nonlinear_solver = om.NonlinearBlockGS()
```

**Linear Solvers (converge derivatives — needed for optimization of coupled systems):**
```python
group.linear_solver = om.DirectSolver()
```

### Beginner Recommendation
For your first coupled model:
```python
model.nonlinear_solver = om.NonlinearBlockGS()
model.nonlinear_solver.options['maxiter'] = 100
model.nonlinear_solver.options['atol'] = 1e-8

model.linear_solver = om.DirectSolver()
```

- `NonlinearBlockGS` is the simplest nonlinear solver — it just runs the components in a loop until values converge
- `DirectSolver` is the simplest linear solver — it works for small-to-medium models
- `maxiter` limits iterations to prevent infinite loops
- `atol` is the absolute tolerance — convergence is achieved when residuals are below this

### Reading Solver Output
Enable solver printing to see what is happening:
```python
model.nonlinear_solver.options['iprint'] = 2
```

Output looks like:
```
NL: NLBGS 0 ; 2.5 1
NL: NLBGS 1 ; 0.8 0.32
NL: NLBGS 2 ; 0.1 0.04
NL: NLBGS 3 ; 0.001 0.0004
NL: NLBGS Converged in 3 iterations
```

- The first number after the semicolon is the absolute residual
- The second number is the relative residual (current / initial)
- You want both to decrease toward zero
- "Converged" means residuals are below tolerance — success!

### When Convergence Fails
If you see:
```
NL: NLBGS Failed to Converge in 100 iterations
```

This means the solver hit `maxiter` without reaching the tolerance.

## Common Fixes for Convergence Issues

### Fix 1: Better Initial Values
The solver starts from whatever values are in the model. If they are far from the solution, convergence is harder.
```python
prob.set_val('some_variable', reasonable_starting_value)
```

### Fix 2: Increase maxiter
Sometimes the solver just needs more iterations:
```python
model.nonlinear_solver.options['maxiter'] = 500
```

### Fix 3: Relax the Tolerance
If the problem is converging but not quite reaching 1e-8, try a looser tolerance to verify the model works:
```python
model.nonlinear_solver.options['atol'] = 1e-6
```
Then tighten once things are working.

### Fix 4: Try a Different Solver
If `NonlinearBlockGS` is not converging, try Newton:
```python
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
model.nonlinear_solver.options['maxiter'] = 20
model.linear_solver = om.DirectSolver()
```
Newton converges faster but requires derivatives.

### Fix 5: Check Your Model
Sometimes convergence failure means the model itself has a problem:
- Are units correct?
- Are connections correct?
- Does the coupled system actually have a solution for these inputs?

## How to Guide the User
- If the user has a simple uncoupled model, tell them they don't need to worry about solvers yet — OpenMDAO handles it automatically
- If the user has coupling, explain it using the A->B->A metaphor
- Start with `NonlinearBlockGS` + `DirectSolver` — don't overwhelm with solver options
- If convergence fails, walk through the five fixes in order
- Always ask: "Is your model coupled? Do any components form a circular dependency?" This determines whether convergence is relevant
- Remind users that solvers are assigned to the Group that contains the coupled components — not to the components themselves
