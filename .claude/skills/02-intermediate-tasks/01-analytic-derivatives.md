# Analytic Derivatives

## Purpose
Teach the user how to implement analytic derivatives using compute_partials(), replacing the finite difference approach used in beginner skills. Analytic derivatives are faster, more accurate, and required for serious optimization work.

## Key Concepts

### Why Analytic Derivatives?
When using method='fd' (finite difference), OpenMDAO approximates derivatives by perturbing inputs and measuring output changes. This is:
- **Slow**: requires extra model evaluations per derivative
- **Inaccurate**: subject to truncation and round-off error
- **Fragile**: step size must be tuned carefully

Analytic derivatives are computed exactly from the math, making optimization faster and more reliable.

### The compute_partials() Method
Add `compute_partials()` alongside `compute()` in your ExplicitComponent:

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Declare exactly which partials exist — no longer using method='fd'
        self.declare_partials('f_xy', 'x')
        self.declare_partials('f_xy', 'y')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        # d(f_xy)/dx = 2*(x-3) + y
        partials['f_xy', 'x'] = 2.0*(x - 3.0) + y

        # d(f_xy)/dy = x + 2*(y+4)
        partials['f_xy', 'y'] = x + 2.0*(y + 4.0)
```

### How to Declare Partials
In `setup_partials()`, declare each output-input pair that has a non-zero derivative:

```python
def setup_partials(self):
    # Declare one at a time
    self.declare_partials('output_name', 'input_name')

    # Or declare multiple inputs for one output
    self.declare_partials('f_xy', ['x', 'y'])

    # Or declare all partials at once (use carefully)
    self.declare_partials('*', '*')
```

### How to Set Partials
In `compute_partials()`, set each declared partial using the (output, input) key:

```python
def compute_partials(self, inputs, partials):
    # Scalar partial
    partials['f_xy', 'x'] = 2.0*(inputs['x'] - 3.0) + inputs['y']

    # If the partial is constant (does not depend on inputs)
    # you can set it once in setup_partials() instead:
    # self.declare_partials('output', 'input', val=constant_value)
```

### Constant Partials
If a partial derivative is constant (does not depend on variable values), declare it directly in `setup_partials()` — no need for `compute_partials()`:

```python
def setup_partials(self):
    # df/dx = 3.0 always — set it once here
    self.declare_partials('f', 'x', val=3.0)
```

### Verifying Your Analytic Derivatives
Always verify with `check_partials()` after implementing:

```python
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid())
prob.setup(force_alloc_complex=True)
prob.set_val('parab.x', 5.0)
prob.set_val('parab.y', -2.0)
prob.run_model()

# Compare analytic vs finite difference
data = prob.check_partials(method='cs', compact_print=True)
```

- Use `method='cs'` (complex step) for the most accurate reference comparison
- `force_alloc_complex=True` in `setup()` is required for complex step
- Look for relative errors below 1e-6 — if larger, your analytic derivative has a bug

### The Sellar Disciplines with Analytic Derivatives

```python
import openmdao.api as om
import numpy as np

class SellarDis1(om.ExplicitComponent):

    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('x', val=0.0)
        self.add_input('y2', val=0.0)
        self.add_output('y1', val=0.0)

    def setup_partials(self):
        self.declare_partials('y1', 'z')
        self.declare_partials('y1', 'x')
        self.declare_partials('y1', 'y2')

    def compute(self, inputs, outputs):
        z = inputs['z']
        x = inputs['x']
        y2 = inputs['y2']
        outputs['y1'] = z[0]**2 + z[1] + x - 0.2*y2

    def compute_partials(self, inputs, partials):
        # dy1/dz = [2*z[0], 1.0]
        partials['y1', 'z'] = np.array([[2.0*inputs['z'][0], 1.0]])

        # dy1/dx = 1.0 (constant — could be declared in setup_partials)
        partials['y1', 'x'] = 1.0

        # dy1/dy2 = -0.2 (constant — could be declared in setup_partials)
        partials['y1', 'y2'] = -0.2


class SellarDis2(om.ExplicitComponent):

    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.0)
        self.add_output('y2', val=0.0)

    def setup_partials(self):
        self.declare_partials('y2', 'z')
        self.declare_partials('y2', 'y1')

    def compute(self, inputs, outputs):
        z = inputs['z']
        y1 = inputs['y1']
        outputs['y2'] = y1**0.5 + z[0] + z[1]

    def compute_partials(self, inputs, partials):
        y1 = inputs['y1']

        # dy2/dz = [1.0, 1.0]
        partials['y2', 'z'] = np.array([[1.0, 1.0]])

        # dy2/dy1 = 0.5 * y1^(-0.5)
        partials['y2', 'y1'] = 0.5 * y1**(-0.5)
```

## Anti-Patterns to Watch For

### Wrong Jacobian Shape for Array Inputs
**Wrong instinct:** Returning a 1D array for a partial involving an array input.
**Why it's wrong:** For an output of size m and input of size n, the Jacobian must be shape (m, n).
**Guide the user to:** Always check shapes. Use `np.array([[...]])` to ensure 2D shape for scalar output, array input.

### Forgetting to Remove method='fd'
**Wrong instinct:** Adding `compute_partials()` but leaving `method='fd'` in `declare_partials()`.
**Why it's wrong:** `method='fd'` overrides `compute_partials()` — analytic derivatives are never used.
**Guide the user to:** Remove `method='fd'` from `declare_partials()` when switching to analytic.

### Not Verifying with check_partials()
**Wrong instinct:** "The math looks right, I don't need to check."
**Guide the user to:** Always run `check_partials()` after implementing analytic derivatives. Derivative bugs are silent — the model runs fine but the optimizer gets wrong gradient information.

## How to Guide the User
- Start by showing the Paraboloid example — it is simple enough to do the math by hand
- Always show `check_partials()` immediately after implementing — make it a habit
- If the user's check_partials() shows large errors, help them debug the math
- For array inputs/outputs, pay careful attention to Jacobian shapes
- Remind the user: constant partials can be declared directly in setup_partials() — no need for compute_partials()
