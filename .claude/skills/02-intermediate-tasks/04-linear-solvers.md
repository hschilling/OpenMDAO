# Linear Solvers

## Purpose
Teach the user what linear solvers do, which ones to use, and how to configure them for the Sellar optimization problem.

## Key Concepts

### What Linear Solvers Do
Linear solvers compute **total derivatives** — the derivatives of outputs with respect to design variables across the entire coupled system. They solve the linear system:

```
[dR/dU] * [dU/dX] = -[dR/dX]
```

This is needed by:
- **Optimizers** — to compute gradients for gradient-based optimization
- **Newton solver** — to compute the Newton step at each iteration

Linear solvers operate on the linearized (Jacobian) form of the model. They are separate from nonlinear solvers.

### The Main Linear Solvers

#### DirectSolver
Assembles the full Jacobian matrix and uses a direct factorization (LU decomposition).

```python
model.linear_solver = om.DirectSolver()
```

**When to use:**
- Small to medium models (up to a few thousand variables)
- Always use with NewtonSolver
- Best starting choice for most beginner/intermediate models

**Limitations:**
- Memory scales as O(n^2) with number of variables
- Compute time scales as O(n^3) — becomes slow for large systems

#### LinearBlockGaussSeidel (LBGS)
Iterative linear solver — solves the linear system block by block.

```python
model.linear_solver = om.LinearBlockGS()
model.linear_solver.options['maxiter'] = 100
model.linear_solver.options['atol'] = 1e-8
```

**When to use:**
- Large models where DirectSolver runs out of memory
- Weakly coupled systems

#### PETScKrylov
Iterative Krylov solver (GMRES by default). Requires PETSc installation.

```python
model.linear_solver = om.PETScKrylov()
model.linear_solver.options['atol'] = 1e-8
```

**When to use:**
- Large-scale problems
- Parallel models

### Sellar Optimization with Linear Solver

```python
import openmdao.api as om
import numpy as np

# (SellarDis1 and SellarDis2 defined as in 03-nonlinear-solvers.md)

class SellarObjective(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('obj', val=0.0)

    def setup_partials(self):
        self.declare_partials('obj', ['x', 'z', 'y1', 'y2'], method='fd')

    def compute(self, inputs, outputs):
        outputs['obj'] = (inputs['x']**2 +
                          inputs['z'][1] +
                          inputs['y1'] +
                          np.exp(-inputs['y2']))


class SellarConstraints(om.ExplicitComponent):

    def setup(self):
        self.add_input('y1', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('con1', val=0.0)
        self.add_output('con2', val=0.0)

    def setup_partials(self):
        self.declare_partials('con1', 'y1')
        self.declare_partials('con2', 'y2')

    def compute(self, inputs, outputs):
        outputs['con1'] = 3.16 - inputs['y1']
        outputs['con2'] = inputs['y2'] - 24.0

    def compute_partials(self, inputs, partials):
        partials['con1', 'y1'] = -1.0
        partials['con2', 'y2'] = 1.0


prob = om.Problem()
model = prob.model

model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])
model.add_subsystem('obj_comp', SellarObjective(), promotes=['x', 'z', 'y1', 'y2', 'obj'])
model.add_subsystem('con_comp', SellarConstraints(), promotes=['y1', 'y2', 'con1', 'con2'])

# Nonlinear solver for coupling between d1 and d2
model.nonlinear_solver = om.NonlinearBlockGS()
model.nonlinear_solver.options['maxiter'] = 100

# Linear solver for total derivatives during optimization
model.linear_solver = om.DirectSolver()

# Driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=0.0, upper=10.0)
prob.model.add_design_var('z', lower=-10.0, upper=10.0)
prob.model.add_objective('obj')
prob.model.add_constraint('con1', upper=0.0)
prob.model.add_constraint('con2', upper=0.0)

prob.setup()
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_driver()

print(f"Optimal x  = {prob.get_val('x')[0]:.6f}")
print(f"Optimal z  = {prob.get_val('z')}")
print(f"Minimum obj = {prob.get_val('obj')[0]:.6f}")
print(f"con1       = {prob.get_val('con1')[0]:.6f}")
print(f"con2       = {prob.get_val('con2')[0]:.6f}")
```

### Linear vs Nonlinear Solver — Summary

| | Nonlinear Solver | Linear Solver |
|--|-----------------|---------------|
| **Purpose** | Converge variable values | Compute total derivatives |
| **When needed** | Coupled systems | Optimization, Newton |
| **Examples** | NLBGS, Newton | DirectSolver, LBGS |
| **Assigned to** | Group | Group |

## Anti-Patterns to Watch For

### Using DirectSolver on Large Models
**Wrong instinct:** "DirectSolver works, so I'll keep using it."
**Why it's wrong:** DirectSolver memory and compute time grow rapidly with model size. For large models it becomes impractical.
**Guide the user to:** For models with thousands of variables, switch to an iterative linear solver like PETScKrylov.

### Mismatched Nonlinear and Linear Solvers
**Wrong instinct:** Using NewtonSolver with LinearBlockGS.
**Why it's wrong:** Newton needs an accurate linear solve at each iteration. LBGS may not converge tightly enough, causing Newton to take wrong steps.
**Guide the user to:** Pair NewtonSolver with DirectSolver for small/medium models.

## How to Guide the User
- If the user is confused about why they need a linear solver, explain: "The optimizer needs to know which direction to move the design variables. That requires computing derivatives across the whole coupled system — that is what the linear solver does."
- Start with DirectSolver — it always works for small/medium models
- Only introduce iterative linear solvers when the user hits memory/performance limits
- Always show nonlinear and linear solver together — they work as a pair
