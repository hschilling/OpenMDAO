# Implicit Components

## Purpose
Teach the user what an ImplicitComponent is, when to use one, and how to implement setup(), setup_partials(), apply_nonlinear(), and linearize() correctly.

## Key Concepts

### What Is an ImplicitComponent?
An ImplicitComponent defines outputs that satisfy a residual equation rather than being computed directly:

- **ExplicitComponent**: `outputs = f(inputs)` — outputs computed directly
- **ImplicitComponent**: `R(inputs, outputs) = 0` — outputs satisfy a residual

Use an ImplicitComponent when:
- Your equation cannot be rearranged to solve explicitly for the output
- You are wrapping an external solver or iterative process
- You want OpenMDAO's Newton solver to converge the output directly

### The Key Methods

```python
class MyImplicitComp(om.ImplicitComponent):

    def setup(self):
        # Declare inputs and outputs — same as ExplicitComponent
        self.add_input('a', val=1.0)
        self.add_output('x', val=0.0)  # x is solved implicitly

    def setup_partials(self):
        self.declare_partials('x', 'a')
        self.declare_partials('x', 'x')  # NOTE: output wrt itself — required

    def apply_nonlinear(self, inputs, outputs, residuals):
        # Define the residual: R = 0 when the system is converged
        # Example: solve x^2 - a = 0, so R = x^2 - a
        residuals['x'] = outputs['x']**2 - inputs['a']

    def linearize(self, inputs, outputs, jacobian):
        # Derivatives of the residual
        # dR/dx = 2*x
        jacobian['x', 'x'] = 2.0 * outputs['x']
        # dR/da = -1.0
        jacobian['x', 'a'] = -1.0
```

### apply_nonlinear() vs compute()
- `apply_nonlinear()` does NOT compute the output — it computes the **residual**
- The residual is how far the current output value is from satisfying the equation
- OpenMDAO's solver drives the residual to zero by iterating on the output value
- When `residuals['x']` is zero, `outputs['x']` holds the converged solution

### linearize() vs compute_partials()
- `linearize()` is the ImplicitComponent equivalent of `compute_partials()`
- It computes derivatives of the **residual** with respect to inputs AND outputs
- Always declare and compute the partial of the residual with respect to the output itself: `jacobian['x', 'x']`

### Simple Example: Square Root via Implicit Formulation
Solve for x such that x^2 = a (i.e., x = sqrt(a)):

```python
import openmdao.api as om

class SqrtImplicit(om.ImplicitComponent):

    def setup(self):
        self.add_input('a', val=4.0)
        self.add_output('x', val=2.0)  # Initial guess matters!

    def setup_partials(self):
        self.declare_partials('x', 'a')
        self.declare_partials('x', 'x')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = outputs['x']**2 - inputs['a']

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = 2.0 * outputs['x']
        jacobian['x', 'a'] = -1.0

prob = om.Problem()
prob.model.add_subsystem('sqrt_comp', SqrtImplicit())

# ImplicitComponents need a solver to converge
prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
prob.model.linear_solver = om.DirectSolver()

prob.setup()
prob.set_val('sqrt_comp.a', 9.0)
prob.run_model()

print(prob.get_val('sqrt_comp.x'))  # Should be 3.0
```

### When You Also Need solve_nonlinear()
By default, OpenMDAO uses the group-level solver to converge ImplicitComponents.
If your component has its own internal solver (e.g., a Newton loop), implement `solve_nonlinear()`:

```python
def solve_nonlinear(self, inputs, outputs):
    # Solve directly — Newton's method for x^2 = a
    outputs['x'] = inputs['a']**0.5
```

If you implement `solve_nonlinear()`, OpenMDAO uses it instead of iterating externally. This is useful for wrapping legacy solvers.

## Anti-Patterns to Watch For

### Confusing Residuals with Outputs
**Wrong instinct:** Setting `outputs['x']` inside `apply_nonlinear()`.
**Why it's wrong:** `apply_nonlinear()` must only write to `residuals`, never to `outputs`. Outputs are managed by the solver.
**Guide the user to:** Think of `apply_nonlinear()` as "evaluate how wrong the current answer is," not "compute the answer."

### Forgetting jacobian['x', 'x']
**Wrong instinct:** Only declaring derivatives of residuals with respect to inputs, not outputs.
**Why it's wrong:** Newton's solver needs dR/d(output) to update the output value. Without it, the solve will fail or converge incorrectly.
**Guide the user to:** Always declare and compute the partial of each residual with respect to its own output variable.

### Bad Initial Guess for the Output
**Wrong instinct:** Leaving the output default value at 0.0.
**Why it's wrong:** Newton's method starts from the current output value. A bad starting point can cause divergence.
**Guide the user to:** Set a physically reasonable initial value for implicit outputs using `val=` in `add_output()` or `prob.set_val()`.

## How to Guide the User
- Start by asking: "Can you rearrange your equation to solve directly for the output?" If yes, use ExplicitComponent. If no, use ImplicitComponent.
- The square root example is the simplest possible ImplicitComponent — use it to illustrate the concept
- Always remind users that ImplicitComponents need a solver at the group level (or their own solve_nonlinear)
- If the user is confused about residuals, use this analogy: "The residual is like a balance scale — it measures how out of balance the equation is. The solver keeps adjusting the output until the scale reads zero."
