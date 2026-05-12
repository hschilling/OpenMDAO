# Custom Solvers

## Purpose
Show how to write and integrate custom nonlinear or linear solvers in OpenMDAO.

## Key Concepts

### Custom Nonlinear Solver
- Inherit from `om.NonlinearSolver`.
- Implement `solve()` and `options` as needed.

```python
import openmdao.api as om

class MyCustomSolver(om.NonlinearSolver):
    def solve(self, system, mode):
        # Custom solve logic
        pass

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('my_option', default=1)
```

### Custom Linear Solver
- Inherit from `om.LinearSolver`.
- Implement `solve()` for linear systems.

```python
class MyCustomLinearSolver(om.LinearSolver):
    def solve(self, system, mode, rhs_vec, sol_vec, tol):
        # Custom linear solve logic
        pass
```

### Integrating Custom Solvers
- Assign your solver to a group or model.

```python
model.nonlinear_solver = MyCustomSolver()
model.linear_solver = MyCustomLinearSolver()
```

## Anti-Patterns to Watch For

- Not implementing required methods: OpenMDAO will raise errors.
- Using custom solvers without proper options or error handling.

## How to Guide the User

- Start from OpenMDAO’s built-in solver templates.
- Test your solver on small models first.
- Document solver options and expected behavior.
