# Your First Component

## Purpose
Teach the user how to create an ExplicitComponent using the Paraboloid example, establishing the foundational pattern they will reuse for every component they build.

## Key Concepts

### What Is an ExplicitComponent?
An ExplicitComponent computes its outputs directly from its inputs. The formula is: outputs = f(inputs). This is the most common component type for beginners.

### The Three Methods You Must Know
Every ExplicitComponent has three key methods:

1. **`setup()`** — Declare your inputs and outputs. This tells OpenMDAO what variables exist, their default values, and their units. This runs once.
2. **`setup_partials()`** — Declare which derivatives exist. This tells OpenMDAO which outputs depend on which inputs. This runs once.
3. **`compute(inputs, outputs)`** — The actual math. Read from `inputs`, write to `outputs`. This runs every time the model executes.

### The Paraboloid Example
The paraboloid is: f(x, y) = (x - 3)^2 + x * y + (y + 4)^2 - 3

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    """A simple paraboloid component."""

    def setup(self):
        # Declare inputs with default values
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        # Declare outputs
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Declare that f_xy depends on both x and y
        # Using method='fd' (finite difference) to start
        self.declare_partials('f_xy', ['x', 'y'], method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
```

### Pattern Rules
- Always import as `import openmdao.api as om`
- Always inherit from `om.ExplicitComponent`
- Always declare ALL inputs and outputs in `setup()`
- Always declare partials in `setup_partials()` — even if using finite difference
- Never read from `outputs` inside `compute()` — only write to it
- Never modify `inputs` inside `compute()` — they are read-only
- Variable names are strings. Keep them descriptive but concise

## Anti-Patterns to Watch For

### Skipping setup_partials()
**Wrong instinct:** "I don't need derivatives, I'm not optimizing."
**Why it's wrong:** Even `run_model()` benefits from properly declared partials. And when the user eventually wants to optimize, they'll have to go back and add them.
**Guide the user to:** Always include `setup_partials()`. It is fine to use `method='fd'` initially to get up and running. Once the model works, encourage replacing finite difference with analytic derivatives for better performance and accuracy.

### Doing Too Much in One Component
**Wrong instinct:** "I'll compute drag, lift, and weight all in one component."
**Guide the user to:** One component, one responsibility. If the computation has logically separable parts, make separate components and connect them.

## How to Guide the User
- If the user asks "how do I create a component," start with this exact pattern
- If the user describes a calculation they want to implement, map it to this template: what are the inputs? What are the outputs? What is the math?
- Always show `setup()`, `setup_partials()`, and `compute()` together — never one without the others
- When the user's component works with `method='fd'`, mention: "Great, this works! When you're ready, you can improve performance by providing analytic derivatives in a `compute_partials()` method."
