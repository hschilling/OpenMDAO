# A/B Test Manual Scoring Sheet

Score each response against the rubric. Award full points or 0 for each criterion.

**DO NOT scroll to the bottom until you have finished scoring** (the answer key is there).

---

## basic-001 (category: basic_concepts)

**Prompt:** What is OpenMDAO?

**Rubric (max 5 points):**
- [ ] mentions_dataflow_graph_or_similar (1 pts)
- [ ] lists_5_building_blocks (2 pts)
- [ ] distinguishes_from_procedural (1 pts)
- [ ] uses_circuit_board_or_similar_metaphor (1 pts)

### Response 1

# What is OpenMDAO?

OpenMDAO is an open-source high-performance computing platform for systems analysis and multidisciplinary design optimization (MDO), written in Python. Here are its key aspects:

## Core Purpose
OpenMDAO is designed to help engineers and researchers:
- **Model complex systems** by connecting multiple disciplinary analysis tools
- **Optimize designs** across multiple disciplines simultaneously
- **Perform trade studies** and sensitivity analysis at scale

## Key Features

### 1. **Modular Framework**
- Allows you to build models from individual components
- Components can be combined hierarchically into groups
- Supports both explicit and implicit components

### 2. **Automatic Derivatives**
- Computes derivatives efficiently for gradient-based optimization
- Supports multiple methods: finite differences, complex step, and automatic differentiation
- Uses advanced algorithms like unified derivatives equation for efficient gradient computation

### 3. **Scalable Architecture**
- Can run on laptops or high-performance computing clusters
- Supports distributed computing with MPI
- Efficient for problems with thousands of design variables

### 4. **Built-in Solvers**
- Nonlinear solvers (Newton, Broyden, nonlinear block Gauss-Seidel)
- Linear solvers (direct and iterative methods)
- Optimization drivers (interfaces to scipy, pyOptSparse, and others)

## Common Applications
- Aircraft and spacecraft design
- Wind turbine optimization
- Automotive systems
- General engineering design optimization

OpenMDAO is developed primarily by NASA Glenn Research Center and is widely used in aerospace and other engineering industries.

**Score Response 1: ___ / 5**

### Response 2

# What is OpenMDAO?

OpenMDAO is an **open-source framework for multidisciplinary design, analysis, and optimization (MDAO)** developed by NASA. It helps engineers and scientists build complex computational models and solve optimization problems where multiple disciplines interact.

## Core Purpose

OpenMDAO solves problems where:
- **Multiple components interact** — outputs from one component feed as inputs to another
- **Feedback loops exist** — component A depends on B, and B depends on A (coupling)
- **You want to optimize** — find the best design by adjusting parameters to minimize/maximize objectives while satisfying constraints

## Key Capabilities

### 1. **Model Building**
Connect computational components together like building blocks:
```python
import openmdao.api as om

prob = om.Problem()
prob.model.add_subsystem('component_A', ComponentA())
prob.model.add_subsystem('component_B', ComponentB())
prob.model.connect('component_A.output', 'component_B.input')
```

### 2. **Automatic Derivative Computation**
OpenMDAO computes derivatives across your entire connected model — essential for gradient-based optimization:
- Handles complex data flow automatically
- Supports analytic derivatives (fast, accurate)
- Provides automatic differentiation as fallback

### 3. **Coupled System Solution**
Solves feedback loops using nonlinear solvers:
- When component A needs output from B, but B needs output from A
- Iterates until all components are mutually consistent

### 4. **Optimization**
Finds optimal designs using gradient-based optimizers:
```python
prob.driver = om.ScipyOptimizeDriver()
prob.model.add_design_var('x', lower=0, upper=10)
prob.model.add_objective('performance')
prob.model.add_constraint('stress', upper=max_stress)
```

## Who Uses OpenMDAO?

- **Aerospace engineers** — aircraft design, trajectory optimization
- **Structural analysts** — weight optimization with stress constraints  
- **Energy researchers** — wind turbine design, power system optimization
- **Anyone with complex coupled systems** to analyze or optimize

## Why OpenMDAO?

### Advantages:
- **Handles complexity** — manages coupling, data flow, and derivatives automatically
- **Scales to large problems** — supports parallel computing and sparse derivatives
- **Open source and extensible** — build custom components, solvers, and drivers
- **Battle-tested** — used in production at NASA and industry

### When NOT to use OpenMDAO:
- Simple uncoupled calculations (overkill)
- Pure data processing with no optimization
- When learning curve outweighs benefit for small problems

## Simple Example

Here's the classic Paraboloid optimization:

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3)**2 + x*y + (y + 4)**2 - 3

# Build and run optimization
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid())

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('parab.x', lower=-10, upper=10)
prob.model.add_design_var('parab.y', lower=-10, upper=10)
prob.model.add_objective('parab.f_xy')

prob.setup()
prob.run_driver()

print(f"Optimal x = {prob.get_val('parab.x')}")
print(f"Optimal y = {prob.get_val('parab.y')}")
print(f"Minimum f = {prob.get_val('parab.f_xy')}")
```

## Bottom Line

**OpenMDAO is for building and optimizing complex computational models where components depend on each other.** If your problem involves coupled physics, feedback loops, or sophisticated optimization with constraints, OpenMDAO provides the infrastructure to solve it efficiently.

Want to dive deeper? I can walk you through building your first model, explain specific features, or help you solve a particular problem!

**Score Response 2: ___ / 5**

---

## basic-002 (category: basic_concepts)

**Prompt:** What is the difference between an ExplicitComponent and an ImplicitComponent?

**Rubric (max 6 points):**
- [ ] explicit_outputs_directly_computed (2 pts)
- [ ] implicit_outputs_satisfy_residual (2 pts)
- [ ] mentions_when_to_use_each (2 pts)

### Response 1

# ExplicitComponent vs ImplicitComponent

The main difference between these two OpenMDAO component types lies in how they compute their outputs:

## **ExplicitComponent**

- **Direct computation**: Outputs are calculated directly from inputs
- **Form**: `y = f(x)`
- **Method required**: `compute()`
- **Use when**: You can explicitly write output as a function of inputs

**Example:**
```python
class ExplicitExample(om.ExplicitComponent):
    def compute(self, inputs, outputs):
        outputs['y'] = 2.0 * inputs['x']**2
```

## **ImplicitComponent**

- **Implicit relationships**: Outputs are defined by residual equations that must equal zero
- **Form**: `R(x, y) = 0` (solver finds y such that residual = 0)
- **Method required**: `apply_nonlinear()` to compute residuals
- **Use when**: 
  - Outputs cannot be easily isolated
  - Circular dependencies exist
  - Solving equations like structural equilibrium, nonlinear circuits, etc.

**Example:**
```python
class ImplicitExample(om.ImplicitComponent):
    def apply_nonlinear(self, inputs, outputs, residuals):
        # Solve: y**2 - 4*x = 0
        residuals['y'] = outputs['y']**2 - 4.0 * inputs['x']
```

## **Key Practical Differences**

| Aspect | Explicit | Implicit |
|--------|----------|----------|
| **Solver needed** | No | Yes (nonlinear solver) |
| **Speed** | Generally faster | Slower (iterative solving) |
| **Derivatives** | `compute_partials()` | `linearize()` for partials |
| **Complexity** | Simpler | More complex setup |

**When in doubt**: Use ExplicitComponent if possible—it's more straightforward and efficient.

**Score Response 1: ___ / 6**

### Response 2

# ExplicitComponent vs ImplicitComponent

The key difference is **how outputs are computed**:

## ExplicitComponent
**Direct computation:** `outputs = f(inputs)`

Outputs are computed directly from inputs in the `compute()` method:

```python
class Paraboloid(om.ExplicitComponent):
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        # Direct calculation
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
```

**Use when:**
- Your equation can be solved explicitly for the output
- You have a direct formula: output = function(inputs)

## ImplicitComponent
**Residual equation:** `R(inputs, outputs) = 0`

Outputs satisfy a residual equation. You define what "converged" means, not how to compute it:

```python
class SqrtImplicit(om.ImplicitComponent):
    
    def apply_nonlinear(self, inputs, outputs, residuals):
        # Define the residual (not the output!)
        # We want x^2 = a, so residual = x^2 - a
        residuals['x'] = outputs['x']**2 - inputs['a']
    
    def linearize(self, inputs, outputs, jacobian):
        # Derivatives of the residual
        jacobian['x', 'x'] = 2.0 * outputs['x']  # dR/dx
        jacobian['x', 'a'] = -1.0                 # dR/da
```

**Use when:**
- Your equation cannot be rearranged to solve explicitly
- You're wrapping an iterative solver
- The relationship is implicit (like x² = a, where x is unknown)

## Key Method Comparison

| Method | ExplicitComponent | ImplicitComponent |
|--------|------------------|-------------------|
| **Compute outputs** | `compute()` | `apply_nonlinear()` (computes residual) |
| **Compute derivatives** | `compute_partials()` | `linearize()` |
| **Needs solver?** | No | Yes (Newton or similar) |

## Critical Difference in apply_nonlinear()

**Wrong (common mistake):**
```python
def apply_nonlinear(self, inputs, outputs, residuals):
    # DON'T compute the output here!
    outputs['x'] = inputs['a']**0.5  # ❌ WRONG
```

**Right:**
```python
def apply_nonlinear(self, inputs, outputs, residuals):
    # Compute the residual - how far from satisfied?
    residuals['x'] = outputs['x']**2 - inputs['a']  # ✅ CORRECT
```

## ImplicitComponent Needs a Solver

```python
prob = om.Problem()
prob.model.add_subsystem('sqrt_comp', SqrtImplicit())

# ImplicitComponents require a solver to converge
prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
prob.model.linear_solver = om.DirectSolver()

prob.setup()
prob.set_val('sqrt_comp.a', 9.0)
prob.run_model()

print(prob.get_val('sqrt_comp.x'))  # 3.0
```

The solver iterates on `outputs['x']` until `residuals['x']` reaches zero.

## When to Use Each?

**Use ExplicitComponent (most common):**
- Direct formulas
- Data transformations
- Most engineering calculations

**Use ImplicitComponent when:**
- Equation form is naturally implicit (balance equations, equilibrium)
- Wrapping an external iterative solver
- Cannot or don't want to rearrange for explicit solution

The Sellar disciplines (SellarDis1, SellarDis2) are ExplicitComponents even though they're coupled - each discipline can compute its own output directly. The coupling is handled by a group-level solver, not by making them ImplicitComponents.

**Score Response 2: ___ / 6**

---

## antipattern-001 (category: anti_pattern_detection)

**Prompt:** I have two components. Component A has output 'temperature' and Component B has input 'temperature'. Both use promotes=['*']. Will they connect correctly, and is this a good practice?

**Rubric (max 9 points):**
- [ ] warns_about_silent_collision (3 pts)
- [ ] recommends_explicit_promotes_lists (2 pts)
- [ ] suggests_list_connections_verification (2 pts)
- [ ] explains_why_dangerous (2 pts)

### Response 1

## Yes, They Will Connect — But With Important Caveats

Your components **will** connect correctly through auto-promotion. When both Component A and Component B promote `'temperature'` to the group level using `promotes=['*']`, OpenMDAO automatically connects A's output to B's input at the group level.

```python
import openmdao.api as om

class CompA(om.ExplicitComponent):
    def setup(self):
        self.add_output('temperature', val=300.0)
    
    def compute(self, inputs, outputs):
        outputs['temperature'] = 350.0

class CompB(om.ExplicitComponent):
    def setup(self):
        self.add_input('temperature', val=0.0)
    
    def compute(self, inputs, outputs):
        print(f"CompB received temperature = {inputs['temperature']}")

prob = om.Problem()
model = prob.model

# Both promote everything — connection happens automatically
model.add_subsystem('comp_a', CompA(), promotes=['*'])
model.add_subsystem('comp_b', CompB(), promotes=['*'])

prob.setup()
prob.run_model()  # CompB will receive 350.0 from CompA
```

## Is This Good Practice?

**It depends on your model complexity:**

### ✅ Good Practice When:
- **Small, simple models** — 2-5 components with few shared variables
- **The variable name is unique and unambiguous** — `'temperature'` only exists once as an output
- **You're prototyping rapidly** — getting a model working quickly

### ⚠️ Risky Practice When:
- **Large models** — many components with many variables
- **Potential naming conflicts** — multiple components might output `'temperature'`
- **You need explicit control** — want to be very clear about what connects where

## The Anti-Pattern to Watch For

### **Wrong Instinct:** "I'll just use `promotes=['*']` everywhere — it's simpler."

**Why it's dangerous:**
1. **Accidental connections** — if another component also outputs `'temperature'`, OpenMDAO will error (you can't promote two outputs to the same name)
2. **Unclear data flow** — in large models, it becomes hard to trace which outputs connect to which inputs
3. **Difficult debugging** — connection errors are harder to diagnose when everything auto-promotes

## Better Practice: Explicit Promotion

For production code or larger models, **explicitly list what you promote:**

```python
# More maintainable — clear what connects where
model.add_subsystem('comp_a', CompA(), promotes_outputs=['temperature'])
model.add_subsystem('comp_b', CompB(), promotes_inputs=['temperature'])
```

Or even more explicit using `connect()`:

```python
# Most explicit — no ambiguity
model.add_subsystem('comp_a', CompA())
model.add_subsystem('comp_b', CompB())
model.connect('comp_a.temperature', 'comp_b.temperature')
```

## When to Use Each Approach

| Approach | Use When |
|----------|----------|
| `promotes=['*']` | Quick prototypes, 2-3 components, learning |
| `promotes=['var1', 'var2']` | Production code, need clarity |
| `connect()` | Complex models, different variable names, explicit control |

## What OpenMDAO Will Tell You

If you have a naming conflict (two components both output `'temperature'`), OpenMDAO will raise an error during `setup()`:

```
RuntimeError: Output name 'temperature' refers to multiple outputs: ['comp_a.temperature', 'comp_c.temperature']
```

This is actually **protective** — OpenMDAO prevents silent connection errors.

## My Recommendation

**For your specific case:** If you only have these two components and `'temperature'` is the only shared variable, `promotes=['*']` is fine.

**As your model grows:** Migrate to explicit promotion lists — your future self (and anyone reading your code) will thank you.

**The golden rule:** *Explicit is better than implicit* — especially in engineering models where correctness is critical.

**Score Response 1: ___ / 9**

### Response 2

# Connection and Best Practices

## Will They Connect?

**Yes**, they will connect automatically. When both components use `promotes=['*']`, OpenMDAO promotes all variables to the group level with their original names. Since both use the name `'temperature'`, the output from Component A and input to Component B will share the same promoted name and connect automatically.

## Is This Good Practice?

**It depends on your model complexity:**

### ✅ **Good for:**
- Small, simple models
- Prototyping and quick scripts
- When variable names are unique and self-documenting

### ⚠️ **Concerns with `promotes=['*']`:**

1. **Name collisions**: If you later add components with similarly named variables, you may get unintended connections or conflicts

2. **Lack of clarity**: It's not immediately obvious which variables are being connected without inspecting component internals

3. **Maintenance issues**: Harder to refactor or debug in larger models

### ✅ **Better Practice (Recommended):**

```python
# Explicit promotion - more maintainable
group.add_subsystem('comp_a', ComponentA(), 
                    promotes_outputs=['temperature'])
group.add_subsystem('comp_b', ComponentB(), 
                    promotes_inputs=['temperature'])
```

Or even better for complex models:

```python
# Use explicit connect statements with unique names
group.add_subsystem('comp_a', ComponentA())
group.add_subsystem('comp_b', ComponentB())
group.connect('comp_a.temperature', 'comp_b.temperature')
```

**Recommendation**: Start with explicit promotion of specific variables, and use `promotes=['*']` only when you're certain about your variable naming scheme and model scope.

**Score Response 2: ___ / 9**

---

## antipattern-002 (category: anti_pattern_detection)

**Prompt:** Should I put my entire aerodynamics analysis in one ExplicitComponent? It is about 500 lines of code.

**Rubric (max 7 points):**
- [ ] recommends_decomposition (3 pts)
- [ ] explains_benefits_of_smaller_components (2 pts)
- [ ] suggests_grouping_related_components (2 pts)

### Response 1

# Should You Split Up a Large Aerodynamics Component?

For a 500-line aerodynamics analysis, **you should probably split it into multiple components**. Here's why and how:

## Reasons to Split:

1. **Modularity & Reusability** - Separate components can be reused in different models
2. **Easier Testing** - Test individual pieces in isolation
3. **Better Derivatives** - Compute partials only where needed; easier to debug
4. **Parallel Computing** - OpenMDAO can potentially parallelize independent components
5. **Clearer Data Flow** - Makes connections and dependencies explicit
6. **Maintenance** - Easier to understand, debug, and modify

## How to Split:

Think about natural breakpoints in your analysis:

```python
# Example: Split aerodynamics into logical pieces

class AtmosphereComp(om.ExplicitComponent):
    """Compute atmospheric properties from altitude"""
    # Inputs: altitude
    # Outputs: rho, temp, pressure, viscosity

class ReynoldsNumberComp(om.ExplicitComponent):
    """Compute Reynolds number"""
    # Inputs: velocity, rho, viscosity, chord
    # Outputs: Re

class LiftCoeffComp(om.ExplicitComponent):
    """Compute lift coefficient"""
    # Inputs: alpha, Re, Mach
    # Outputs: CL

class DragCoeffComp(om.ExplicitComponent):
    """Compute drag coefficient"""
    # Inputs: CL, Re, Mach
    # Outputs: CD

class AeroForcesComp(om.ExplicitComponent):
    """Compute forces from coefficients"""
    # Inputs: CL, CD, q, S
    # Outputs: L, D
```

## When to Keep Together:

Keep it as **one component** if:
- The computations are tightly coupled with expensive shared calculations
- You're calling an external code/library that returns everything at once
- Splitting would require redundant computations

## Best Practice:

Use a **Group** to organize related components:

```python
class AerodynamicsGroup(om.Group):
    def setup(self):
        self.add_subsystem('atmosphere', AtmosphereComp())
        self.add_subsystem('reynolds', ReynoldsNumberComp())
        self.add_subsystem('aero_coeffs', AeroCoeffsComp())
        self.add_subsystem('forces', AeroForcesComp())
        
        # Connect them
        self.connect('atmosphere.rho', 'reynolds.rho')
        # etc...
```

This gives you modularity while keeping the interface clean for users of your aerodynamics analysis.

**Score Response 1: ___ / 7**

### Response 2

## No — Break It Into Multiple Components

Putting 500 lines of aerodynamics analysis into one component is an **anti-pattern**. Here's why and how to structure it better:

### Problems with One Large Component

1. **Hard to maintain and debug** — A 500-line `compute()` method is difficult to understand, test, and modify

2. **Difficult to verify derivatives** — `check_partials()` on one huge component gives you one pass/fail result. You won't know which specific calculation has the derivative bug.

3. **Prevents selective execution** — If you only need to update part of the analysis, you have to re-run all 500 lines

4. **Makes the model structure unclear** — Other users (including future you) won't understand the data flow

### Better Approach: Component Decomposition

Break your aerodynamics analysis into a **Group** containing multiple smaller components that represent distinct physical calculations:

```python
import openmdao.api as om

class AerodynamicsGroup(om.Group):
    
    def setup(self):
        # Break analysis into logical chunks
        self.add_subsystem('geometry',
                           GeometryCalculations(),
                           promotes_inputs=['wing_area', 'aspect_ratio'])
        
        self.add_subsystem('lift',
                           LiftCalculations(),
                           promotes_inputs=['alpha', 'velocity'])
        
        self.add_subsystem('drag',
                           DragCalculations(),
                           promotes_inputs=['velocity'])
        
        self.add_subsystem('moments',
                           MomentCalculations())
        
        # Connect intermediate variables between components
        self.connect('geometry.wing_span', 'lift.wing_span')
        self.connect('geometry.chord', ['drag.chord', 'moments.chord'])
        self.connect('lift.CL', 'drag.induced_drag_factor')
```

### Guidelines for Component Size

**Good component:**
- Represents one physical calculation (lift, drag, moment)
- 50-150 lines of code in `compute()`
- Has clear inputs and outputs
- Can be tested and verified independently

**Too small:**
- Just one arithmetic operation → use ExecComp instead
- Creating unnecessary data passing overhead

**Too large:**
- Multiple distinct physical calculations mixed together
- Hard to understand the data flow
- Difficult to debug derivatives

### Example: Breaking Down Aerodynamics

If your 500-line analysis computes lift, drag, and moments, structure it like this:

```python
class LiftCalculation(om.ExplicitComponent):
    """
    Computes lift coefficient and lift force.
    
    Inputs: alpha (angle of attack), velocity, wing_area
    Outputs: CL, lift_force
    """
    
    def setup(self):
        self.add_input('alpha', val=0.0, units='rad')
        self.add_input('velocity', val=0.0, units='m/s')
        self.add_input('wing_area', val=0.0, units='m**2')
        
        self.add_output('CL', val=0.0)
        self.add_output('lift_force', val=0.0, units='N')
    
    def setup_partials(self):
        self.declare_partials('*', '*')
    
    def compute(self, inputs, outputs):
        # ~50 lines of lift calculation
        alpha = inputs['alpha']
        # ... lift coefficient calculation ...
        outputs['CL'] = CL_computed
        outputs['lift_force'] = 0.5 * rho * V**2 * S * CL_computed
    
    def compute_partials(self, inputs, partials):
        # Derivatives for just the lift calculation
        # Much easier to verify than 500-line component
        pass


class DragCalculation(om.ExplicitComponent):
    """
    Computes drag coefficient and drag force.
    
    Inputs: CL, velocity, wing_area, form_factors
    Outputs: CD, drag_force
    """
    # Similar structure...
```

### Benefits of This Approach

✅ **Each component can be tested independently**
```python
# Test just the lift calculation
prob = om.Problem()
prob.model.add_subsystem('lift', LiftCalculation())
prob.setup(force_alloc_complex=True)
data = prob.check_partials(method='cs')
```

✅ **Clear data flow** — connections show which calculations depend on others

✅ **Easier debugging** — if optimization fails, you can isolate which component's derivatives are wrong

✅ **Reusable** — You can use LiftCalculation in other models without copying 500 lines

✅ **Better version control** — Changes to drag calculation don't affect lift calculation diffs

### When One Component Is OK

**Single component is fine when:**
- The calculation is inherently monolithic (can't be meaningfully subdivided)
- Intermediate values have no physical meaning outside the component
- The total is under ~150 lines

**Example of appropriate single component:** A proprietary aerodynamics correlation that takes 10 inputs and produces 3 outputs via a complex internal calculation.

### Summary

Your 500-line aerodynamics analysis should be a **Group containing 3-6 components**, each representing a distinct physical calculation. This makes the model easier to understand, test, debug, and maintain.

The time spent organizing components upfront pays back many times over when you need to verify derivatives, debug optimization failures, or modify the analysis later.

**Score Response 2: ___ / 7**

---

## antipattern-003 (category: anti_pattern_detection)

**Prompt:** I do not need analytic derivatives because I am only running the model once, not optimizing. Can I skip declaring partials entirely?

**Rubric (max 6 points):**
- [ ] advises_against_skipping_setup_partials (2 pts)
- [ ] recommends_method_fd_as_starting_point (2 pts)
- [ ] mentions_future_optimization_compatibility (2 pts)

### Response 1

# Short Answer

**Yes**, if you're only running the model once (not optimizing), you can skip declaring partials entirely. OpenMDAO won't compute derivatives unless you explicitly request them.

# Important Details

However, there are some nuances to consider:

## 1. **No Warning by Default**
If you don't declare partials and then try to compute derivatives later, OpenMDAO will assume all partials are zero (unless you have `declare_coloring()` or other settings). You'll get a warning like:
```
"Component 'xxx' has not declared partials for output 'yyy' with respect to input 'zzz'"
```

## 2. **Best Practice - Be Explicit**
Even if not optimizing now, you might later. Consider:

```python
class MyComp(om.ExplicitComponent):
    def setup(self):
        # ... inputs/outputs ...
        
        # If you know you'll NEVER need derivatives:
        # Just don't declare anything
        
        # OR be explicit that you don't want to provide them:
        self.declare_partials('*', '*', method='fd')  # Use finite difference if needed
```

## 3. **When You Can Safely Skip**
- Running a single forward execution
- Using the model for evaluation only
- Using `problem.run_model()` without `problem.compute_totals()` or optimization

## 4. **When You Shouldn't Skip**
- Using an optimizer/driver (DOEDriver is an exception)
- Calling `compute_totals()` or `check_totals()`
- Planning to use the model for gradient-based optimization later

# Bottom Line

For pure forward execution, skipping `declare_partials()` is perfectly fine and won't cause errors.

**Score Response 1: ___ / 6**

### Response 2

No, you cannot skip declaring partials entirely. Here's what actually happens:

## What Happens If You Skip Partials

If you add an `ExplicitComponent` without any `declare_partials()` calls, OpenMDAO will **raise an error** when you try to compute derivatives (like during optimization or when calling `check_partials()`).

However, if you are **truly only running the model once** (just `prob.run_model()` with no optimization, no `check_partials()`, no `check_totals()`), then technically the partials are never needed and the code will run.

## The Right Way to Think About This

Even if you're not optimizing **right now**, follow this guidance:

### For Rapid Prototyping (temporary code)
Use finite difference as a placeholder:

```python
def setup_partials(self):
    # Quick placeholder — will work but slow for optimization later
    self.declare_partials('*', '*', method='fd')
```

This lets you:
- Run the model immediately
- Call `check_partials()` to debug if needed
- Switch to optimization later without errors

### For Production Code (even if not optimizing yet)
Always implement analytic derivatives from the start:

```python
def setup_partials(self):
    self.declare_partials('output', 'input')

def compute_partials(self, inputs, partials):
    partials['output', 'input'] = # ... your analytic derivative
```

**Why?** Because:
1. You will almost always want to optimize eventually
2. Other users may connect your component to an optimization
3. Derivatives are how you verify your component math is correct (`check_partials()` is a powerful debugging tool even without optimization)

## The Anti-Pattern

**Wrong instinct:** "I'm not optimizing, so I'll just skip all the derivative stuff."

**Why it's wrong:** 
- Your component becomes unusable in any optimization context
- You lose the debugging power of `check_partials()`
- When you do need optimization later, you have to retrofit derivatives (much harder than doing it upfront)

**Guide yourself to:** At minimum, declare finite difference derivatives. For any component you plan to keep, implement analytic derivatives from the start.

## Bottom Line

If you're literally running the model once and throwing away the code: skip it.

If you're building something that will live longer than today: add `declare_partials('*', '*', method='fd')` at minimum, and plan to implement analytic derivatives before serious use.

**Score Response 2: ___ / 6**

---

## convergence-001 (category: convergence_debugging)

**Prompt:** My solver says 'NL: NLBGS Failed to Converge in 100 iterations'. What do I do?

**Rubric (max 7 points):**
- [ ] suggests_better_initial_values (1 pts)
- [ ] suggests_increase_maxiter (1 pts)
- [ ] suggests_relax_tolerance (1 pts)
- [ ] suggests_try_newton_solver (2 pts)
- [ ] suggests_check_model_correctness (1 pts)
- [ ] presents_in_order_of_complexity (1 pts)

### Response 1

When NLBGS (Nonlinear Block Gauss-Seidel) fails to converge, here are steps to diagnose and fix the issue:

## 1. **Increase Maximum Iterations**
```python
nlbgs = model.nonlinear_solver = om.NonlinearBlockGS()
nlbgs.options['maxiter'] = 200  # or higher
```

## 2. **Tighten or Loosen Tolerances**
```python
nlbgs.options['atol'] = 1e-6  # absolute tolerance
nlbgs.options['rtol'] = 1e-6  # relative tolerance
```

## 3. **Add Relaxation (Under-relaxation)**
This can help with stability:
```python
nlbgs.options['use_aitken'] = True  # Aitken relaxation
# or
nlbgs.options['aitken_initial_factor'] = 0.5  # more conservative
```

## 4. **Try a Different Solver**
NLBGS can struggle with tight coupling. Consider:
```python
# Newton solver (more robust for tight coupling)
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
model.linear_solver = om.DirectSolver()
```

## 5. **Check Your Problem Setup**
- **Bad initial guesses**: Set better initial values for states
- **Scaling issues**: Add `ref` and `ref0` to normalize variables
- **Circular dependencies**: Check for algebraic loops

## 6. **Debug with Print Options**
```python
nlbgs.options['iprint'] = 2  # print convergence info
nlbgs.options['err_on_non_converge'] = False  # continue despite non-convergence
```

## 7. **Visualize the Problem**
```python
om.n2(model, show_browser=True)  # check for coupling
```

**Most common fix**: Switch to Newton solver if you have tight coupling between components, or improve initial guesses for your states.

What type of problem are you solving? That might help narrow down the best solution.

**Score Response 1: ___ / 7**

### Response 2

# Debugging NLBGS Convergence Failure

When you see "NLBGS Failed to Converge in 100 iterations", your nonlinear solver is not finding a consistent solution for your coupled system. Here's how to diagnose and fix it:

## Step 1: Understand What's Happening

NLBGS (NonlinearBlockGaussSeidel) iterates through your coupled components one at a time, passing outputs to the next component's inputs. It converges when the feedback variables stop changing. If it hits `maxiter` without converging, the coupling is too strong for this iterative approach.

## Step 2: Check Your Residuals

Increase the verbosity to see what's actually happening:

```python
model.nonlinear_solver = om.NonlinearBlockGS()
model.nonlinear_solver.options['maxiter'] = 100
model.nonlinear_solver.options['iprint'] = 2  # Print each iteration
model.nonlinear_solver.options['atol'] = 1e-8
model.nonlinear_solver.options['rtol'] = 1e-8
```

Run your model and look at the printed residual norms. Are they:
- **Decreasing but slowly?** → Increase `maxiter` or switch to Newton
- **Oscillating?** → The coupling is too tight — switch to Newton
- **Exploding (getting larger)?** → Bad initial values or unstable physics

## Step 3: Check Your Initial Values

Bad starting points are a common cause of non-convergence:

```python
# Before prob.setup() or prob.run_model()
prob.set_val('y1', 1.0)  # Set reasonable initial guess
prob.set_val('y2', 1.0)
```

For the Sellar problem, starting with `y1=y2=0` can cause issues because `y2 = sqrt(y1)` — you can't take the square root of a negative number if the iterations drive y1 negative.

## Step 4: Switch to NewtonSolver

If NLBGS is not converging, your system is likely **strongly coupled** — meaning the feedback variables have a large effect on each other. Newton's method solves the entire coupled system simultaneously:

```python
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
model.nonlinear_solver.options['maxiter'] = 20  # Newton needs fewer iterations
model.nonlinear_solver.options['atol'] = 1e-8
model.nonlinear_solver.options['rtol'] = 1e-8
model.nonlinear_solver.options['iprint'] = 2

# Newton REQUIRES a linear solver
model.linear_solver = om.DirectSolver()
```

Newton is more robust for tight coupling but requires:
- A linear solver (always use `DirectSolver` for small/medium models)
- Analytic derivatives (or at least `method='fd'` on `declare_partials()`)

## Step 5: Relax Tolerances (Temporary Diagnostic)

To see if you're close to converging, try relaxing the tolerances:

```python
model.nonlinear_solver.options['atol'] = 1e-6  # Less strict
model.nonlinear_solver.options['rtol'] = 1e-6
```

If it now converges, you know the solver is working — you just need more iterations or a better algorithm. Tighten the tolerances back afterward.

## Step 6: Check for Missing Connections

Sometimes NLBGS fails because a connection is missing and variables are not actually coupled:

```python
# After prob.setup()
om.n2(prob)  # Open the N2 diagram in a browser
```

Look for unconnected inputs or broken feedback paths.

## Complete Working Example: Sellar with Newton

```python
import openmdao.api as om
import numpy as np

class SellarDis1(om.ExplicitComponent):
    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('x', val=0.0)
        self.add_input('y2', val=1.0)  # Good initial value
        self.add_output('y1', val=1.0)
    
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
    
    def compute(self, inputs, outputs):
        z = inputs['z']
        outputs['y1'] = z[0]**2 + z[1] + inputs['x'] - 0.2*inputs['y2']

class SellarDis2(om.ExplicitComponent):
    def setup(self):
        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=1.0)  # Good initial value
        self.add_output('y2', val=1.0)
    
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
    
    def compute(self, inputs, outputs):
        outputs['y2'] = inputs['y1']**0.5 + inputs['z'][0] + inputs['z'][1]

prob = om.Problem()
model = prob.model

model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

# Use Newton instead of NLBGS
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
model.nonlinear_solver.options['maxiter'] = 20
model.nonlinear_solver.options['iprint'] = 2
model.linear_solver = om.DirectSolver()

prob.setup()
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

prob.run_model()

print(f"y1 = {prob.get_val('y1')}")
print(f"y2 = {prob.get_val('y2')}")
```

## Quick Decision Tree

```
NLBGS not converging?
│
├─ Are residuals decreasing slowly?
│  → Increase maxiter to 200 or switch to Newton
│
├─ Are residuals oscillating?
│  → Switch to Newton (tight coupling)
│
├─ Are residuals exploding?
│  → Check initial values, check physics for stability
│
└─ Did you just add/change a component?
   → Check connections with om.n2(prob)
```

## Common Anti-Pattern: Giving Up on NLBGS Too Early

**Wrong instinct:** "NLBGS failed at 100 iterations, I need to completely rethink my model."

**Reality:** Try Newton first. Many coupled systems need simultaneous solution. It's not a failure of your model — it's just that NLBGS is the wrong tool for that particular coupling strength.

Let me know what you see when you increase `iprint=2` — the residual history will tell us exactly what's going wrong!

**Score Response 2: ___ / 7**

---

## convergence-002 (category: convergence_debugging)

**Prompt:** My optimizer is taking 500 iterations and not converging. My design variable ranges from 0 to 100,000.

**Rubric (max 8 points):**
- [ ] identifies_scaling_as_likely_cause (3 pts)
- [ ] recommends_ref_value_on_design_var (2 pts)
- [ ] explains_order_1_rule (2 pts)
- [ ] shows_correct_add_design_var_with_ref (1 pts)

### Response 1

# Diagnosis: Scaling Problem

Your optimizer is struggling because your design variable is **extremely poorly scaled**. Gradient-based optimizers work best when all variables are **order 1** (roughly between 0.1 and 10). With a range of 0 to 100,000, your optimizer is seeing:

- Tiny gradients (derivatives look like 0.00001 or smaller)
- Steps that are either too large or too small
- Numerical conditioning issues

This is a classic scaling problem.

## Immediate Fix

Add scaling to your design variable using `ref`:

```python
# Bad — optimizer sees values from 0 to 100,000
prob.model.add_design_var('my_var', lower=0.0, upper=100000.0)

# Good — optimizer sees values from 0 to 1
prob.model.add_design_var('my_var', 
                          lower=0.0, 
                          upper=100000.0,
                          ref=100000.0,    # 100,000 maps to 1.0 for optimizer
                          ref0=0.0)        # 0 maps to 0.0 for optimizer
```

**The physics stays the same** — scaling only affects what the optimizer sees. Your component still computes with the physical values.

## How Scaling Works

OpenMDAO transforms variables for the optimizer:
```
optimizer_value = (physical_value - ref0) / (ref - ref0)
```

So when your physical variable is 50,000:
- Without scaling: optimizer sees 50,000 (very large)
- With `ref=100000, ref0=0`: optimizer sees 0.5 (order 1 — perfect!)

## Check Your Objective and Constraints Too

Design variables are usually not the only problem. Check the magnitude of your objective and constraints:

```python
# Run the model at a typical point
prob.run_model()

# Print the values
print(f"Objective value: {prob.get_val('obj')}")
print(f"Constraint values: {prob.get_val('con1')}, {prob.get_val('con2')}")
```

**Apply scaling to anything that is not order 1:**

```python
# If objective is 1e-6 (too small)
prob.model.add_objective('obj', ref=1e-6)  # Scales it up to order 1

# If constraint upper bound is 1e8 (too large)
prob.model.add_constraint('stress', upper=1e8, ref=1e8)  # Scales to order 1
```

## After Adding Scaling

1. **Verify your derivatives are still correct:**
```python
prob.run_model()
prob.check_totals(method='cs', compact_print=True)
```

2. **Run the optimizer again** — you should see:
   - Convergence in 10-50 iterations (not 500)
   - Meaningful progress each iteration
   - The optimizer respecting your bounds

## The Golden Rule

**All design variables, objectives, and constraints should be approximately order 1 from the optimizer's perspective.**

- Set `ref` to the typical or maximum value of each variable
- For a range [0, 100000], use `ref=100000`
- For a range [-1e6, 1e6], use `ref=1e6` (or use `scaler=1e-6` for simplicity)

Would you like me to help you add scaling to your specific model? Please share:
- What your design variable represents (so we can choose sensible ref values)
- The typical magnitude of your objective
- The upper/lower bounds on your constraints

**Score Response 1: ___ / 8**

### Response 2

# Scaling Issues in OpenMDAO Optimization

Your convergence problem is likely due to **poor scaling**. A design variable ranging from 0 to 100,000 can cause numerical issues for most optimizers, which work best when variables are O(1).

## Solutions:

### 1. **Add Scaling to Your Design Variable** (Recommended)

```python
# Scale the design variable to roughly 0-1 range
prob.model.add_design_var('x', lower=0, upper=100000, 
                          ref=100000, ref0=0)
```

Or use `scaler` for simple scaling:
```python
prob.model.add_design_var('x', lower=0, upper=100000, 
                          scaler=1e-5)  # Divides by 100,000
```

### 2. **Scale Your Objectives and Constraints Too**

```python
prob.model.add_objective('obj', ref=1000)  # if obj is O(1000)
prob.model.add_constraint('con', lower=0, upper=1e6, 
                         ref=1e6)  # if constraint is O(1e6)
```

### 3. **Check Your Scaling Quality**

After setup, run:
```python
prob.setup()
prob.run_model()
prob.driver.scaling_report()
```

Look for variables with magnitudes far from 1.0. Aim for all scaled values to be between 0.1 and 10.

### 4. **Additional Tips**

- Try tighter convergence tolerances on your driver:
  ```python
  prob.driver.options['tol'] = 1e-8
  ```
- Consider different optimizers (SLSQP vs SNOPT vs IPOPT)
- Check if your objective is properly defined and derivatives are correct

**What optimizer are you using?** Some are more sensitive to scaling than others.

**Score Response 2: ___ / 8**

---

## code-gen-001 (category: code_generation_simple)

**Prompt:** Create a paraboloid component.

**Rubric (max 6 points):**
- [ ] uses_om_alias (1 pts)
- [ ] inherits_explicit_component (1 pts)
- [ ] has_setup_method (1 pts)
- [ ] has_setup_partials_method (1 pts)
- [ ] has_compute_method (1 pts)
- [ ] correct_paraboloid_formula (1 pts)

### Response 1

I'll help you create a paraboloid component in OpenMDAO. Here's a complete example:

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
    """

    def setup(self):
        # Define inputs
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        # Define outputs
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Declare derivatives
        self.declare_partials('f_xy', 'x')
        self.declare_partials('f_xy', 'y')

    def compute(self, inputs, outputs):
        """
        Evaluate the paraboloid function.
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """
        Calculate partial derivatives.
        """
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0 * (x - 3.0) + y
        partials['f_xy', 'y'] = x + 2.0 * (y + 4.0)


# Example usage
if __name__ == '__main__':
    # Build and run a simple model
    model = om.Group()
    model.add_subsystem('paraboloid', Paraboloid())

    prob = om.Problem(model)
    prob.setup()

    # Set input values
    prob.set_val('paraboloid.x', 3.0)
    prob.set_val('paraboloid.y', -4.0)

    # Run the model
    prob.run_model()

    # Print the output
    print(f"f_xy = {prob.get_val('paraboloid.f_xy')[0]}")
```

This component:
- **Inputs**: `x` and `y` (design variables)
- **Output**: `f_xy` (the paraboloid function value)
- **Derivatives**: Analytical derivatives are provided for gradient-based optimization

The paraboloid has a minimum at `x=3, y=-4` where `f_xy = -3`.

Would you like to see how to use this in an optimization problem?

**Score Response 1: ___ / 6**

### Response 2

I'll help you create a paraboloid component in OpenMDAO. Here's a complete, well-structured implementation:

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        # Declare inputs
        self.add_input('x', val=0.0, desc='x coordinate')
        self.add_input('y', val=0.0, desc='y coordinate')
        
        # Declare output
        self.add_output('f_xy', val=0.0, desc='function value')

    def setup_partials(self):
        # Declare analytic derivatives
        self.declare_partials('f_xy', 'x')
        self.declare_partials('f_xy', 'y')

    def compute(self, inputs, outputs):
        """Evaluate the paraboloid function."""
        x = inputs['x']
        y = inputs['y']
        
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """Compute analytic derivatives."""
        x = inputs['x']
        y = inputs['y']
        
        # df/dx = 2*(x-3) + y
        partials['f_xy', 'x'] = 2.0 * (x - 3.0) + y
        
        # df/dy = x + 2*(y+4)
        partials['f_xy', 'y'] = x + 2.0 * (y + 4.0)


# Example usage
if __name__ == '__main__':
    import numpy as np
    
    # Create problem
    prob = om.Problem()
    prob.model.add_subsystem('parab', Paraboloid(),
                             promotes_inputs=['x', 'y'],
                             promotes_outputs=['f_xy'])
    
    # Set up with complex step support for derivative checking
    prob.setup(force_alloc_complex=True)
    
    # Set input values
    prob.set_val('x', 3.0)
    prob.set_val('y', -4.0)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print(f"f({prob.get_val('x')[0]}, {prob.get_val('y')[0]}) = {prob.get_val('f_xy')[0]}")
    
    # Verify derivatives
    print("\nChecking analytic derivatives:")
    data = prob.check_partials(method='cs', compact_print=True)
```

## Key Features:

1. **Analytic Derivatives**: Uses `compute_partials()` for fast, accurate derivatives instead of finite difference
2. **Proper Setup**: Declares inputs and outputs in `setup()`
3. **Derivative Declaration**: Uses `setup_partials()` to declare which partials exist
4. **Verification**: Includes `check_partials()` to verify the analytic derivatives

## Expected Output:
```
f(3.0, -4.0) = -3.0

Checking analytic derivatives:
--------------------------------
Component: Paraboloid 'parab'
--------------------------------
'parab.f_xy' wrt 'parab.x'
  Analytic Magnitude: 2.000e+00
  Fd Magnitude      : 2.000e+00
  Absolute Error    : 0.000e+00
  Relative Error    : 0.000e+00    OK

'parab.f_xy' wrt 'parab.y'
  Analytic Magnitude: 3.000e+00
  Fd Magnitude      : 3.000e+00
  Absolute Error    : 0.000e+00
  Relative Error    : 0.000e+00    OK
```

The minimum of this paraboloid is at (x=6.6667, y=-7.3333) with f_xy=-27.3333. Would you like me to show you how to find this minimum using optimization?

**Score Response 2: ___ / 6**

---

## code-gen-002 (category: code_generation_constrained)

**Prompt:** Create a component for lift force using L = 0.5 * rho * V^2 * S * Cl. Make sure I can use this in optimization later.

**Rubric (max 8 points):**
- [ ] includes_units_on_physical_quantities (2 pts)
- [ ] uses_descriptive_variable_names (1 pts)
- [ ] includes_setup_partials (2 pts)
- [ ] mentions_analytic_derivatives_for_optimization (2 pts)
- [ ] warns_about_fd_for_optimization (1 pts)

### Response 1

# Lift Force Component

Here's a properly structured component for lift force calculation, ready for optimization:

```python
import openmdao.api as om
import numpy as np


class LiftForce(om.ExplicitComponent):
    """
    Computes lift force using the standard aerodynamic equation:
    L = 0.5 * rho * V^2 * S * Cl
    
    Inputs:
        rho : air density (kg/m^3)
        V : velocity (m/s)
        S : reference area (m^2)
        Cl : lift coefficient (dimensionless)
    
    Outputs:
        L : lift force (N)
    """

    def setup(self):
        # Declare inputs with units
        self.add_input('rho', val=1.225, units='kg/m**3', 
                       desc='air density')
        self.add_input('V', val=50.0, units='m/s', 
                       desc='velocity')
        self.add_input('S', val=10.0, units='m**2', 
                       desc='reference area')
        self.add_input('Cl', val=0.5, 
                       desc='lift coefficient')
        
        # Declare output with units
        self.add_output('L', val=0.0, units='N', 
                        desc='lift force')

    def setup_partials(self):
        # Declare all partials - needed for optimization
        self.declare_partials('L', ['rho', 'V', 'S', 'Cl'])

    def compute(self, inputs, outputs):
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        Cl = inputs['Cl']
        
        # L = 0.5 * rho * V^2 * S * Cl
        outputs['L'] = 0.5 * rho * V**2 * S * Cl

    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        Cl = inputs['Cl']
        
        # dL/drho = 0.5 * V^2 * S * Cl
        partials['L', 'rho'] = 0.5 * V**2 * S * Cl
        
        # dL/dV = 0.5 * rho * 2*V * S * Cl = rho * V * S * Cl
        partials['L', 'V'] = rho * V * S * Cl
        
        # dL/dS = 0.5 * rho * V^2 * Cl
        partials['L', 'S'] = 0.5 * rho * V**2 * Cl
        
        # dL/dCl = 0.5 * rho * V^2 * S
        partials['L', 'Cl'] = 0.5 * rho * V**2 * S


# Example usage and verification
if __name__ == "__main__":
    prob = om.Problem()
    
    prob.model.add_subsystem('lift', LiftForce(),
                            promotes=['*'])
    
    # Enable complex step for accurate derivative checking
    prob.setup(force_alloc_complex=True)
    
    # Set typical cruise conditions
    prob.set_val('rho', 1.225)      # sea level density
    prob.set_val('V', 50.0)         # 50 m/s
    prob.set_val('S', 10.0)         # 10 m^2 wing area
    prob.set_val('Cl', 0.5)         # typical cruise Cl
    
    prob.run_model()
    
    print(f"Lift force: {prob.get_val('L')[0]:.2f} N")
    
    # ALWAYS verify derivatives before optimization
    print("\nVerifying analytic derivatives...")
    data = prob.check_partials(method='cs', compact_print=True)
    
    print("\nAll derivatives verified! Ready for optimization.")
```

## Key Features for Optimization

1. **Analytic derivatives** — `compute_partials()` provides exact gradients, much faster than finite difference

2. **Proper units** — All variables have physical units declared, allowing OpenMDAO to handle conversions automatically

3. **All partials declared** — Every input-output derivative pair is specified in `setup_partials()`

4. **Verified** — `check_partials()` confirms derivatives are correct before optimization

## Example Optimization Setup

Here's how you'd use this in an optimization:

```python
import openmdao.api as om

prob = om.Problem()

# Add the lift component
prob.model.add_subsystem('lift', LiftForce(),
                        promotes=['*'])

# Set up optimization driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# Example: minimize Cl while maintaining required lift
prob.model.add_design_var('Cl', lower=0.1, upper=2.0)
prob.model.add_design_var('V', lower=20.0, upper=100.0)

prob.model.add_objective('Cl')  # minimize lift coefficient

# Constraint: must generate at least 5000 N of lift
prob.model.add_constraint('L', lower=5000.0)

# Add recorder to track optimization
recorder = om.SqliteRecorder('lift_opt.sql')
prob.driver.add_recorder(recorder)

prob.setup()

# Set initial values
prob.set_val('rho', 1.225)
prob.set_val('V', 50.0)
prob.set_val('S', 10.0)
prob.set_val('Cl', 0.5)

# Verify total derivatives before optimizing
prob.run_model()
prob.check_totals(compact_print=True)

# Run optimization
prob.run_driver()

print(f"\nOptimal Cl: {prob.get_val('Cl')[0]:.4f}")
print(f"Optimal V: {prob.get_val('V')[0]:.2f} m/s")
print(f"Final lift: {prob.get_val('L')[0]:.2f} N")

prob.cleanup()
```

## Alternative: Quick Prototyping with ExecComp

For rapid prototyping, you could also use ExecComp:

```python
prob.model.add_subsystem('lift',
    om.ExecComp('L = 0.5 * rho * V**2 * S * Cl',
                rho={'val': 1.225, 'units': 'kg/m**3'},
                V={'val': 50.0, 'units': 'm/s'},
                S={'val': 10.0, 'units': 'm**2'},
                Cl={'val': 0.5},
                L={'units': 'N'}),
    promotes=['*'])
```

**However**, the explicit component with analytic derivatives will be faster and more accurate for optimization. Use ExecComp for quick tests, then graduate to a full component for production work.

**Score Response 1: ___ / 8**

### Response 2

Here's a component for calculating lift force that's optimization-ready:

```python
import openmdao.api as om

class LiftForceComp(om.ExplicitComponent):
    """
    Computes lift force using: L = 0.5 * rho * V^2 * S * Cl
    
    Inputs:
        rho : air density (kg/m^3)
        V : velocity (m/s)
        S : wing reference area (m^2)
        Cl : lift coefficient (dimensionless)
    
    Outputs:
        L : lift force (N)
    """
    
    def setup(self):
        # Inputs
        self.add_input('rho', val=1.225, units='kg/m**3', 
                       desc='Air density')
        self.add_input('V', val=50.0, units='m/s', 
                       desc='Airspeed')
        self.add_input('S', val=10.0, units='m**2', 
                       desc='Wing reference area')
        self.add_input('Cl', val=0.5, 
                       desc='Lift coefficient')
        
        # Output
        self.add_output('L', val=1.0, units='N', 
                        desc='Lift force')
        
        # Declare partials for optimization
        self.declare_partials('L', ['rho', 'V', 'S', 'Cl'])
    
    def compute(self, inputs, outputs):
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        Cl = inputs['Cl']
        
        outputs['L'] = 0.5 * rho * V**2 * S * Cl
    
    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        Cl = inputs['Cl']
        
        # Analytical derivatives
        partials['L', 'rho'] = 0.5 * V**2 * S * Cl
        partials['L', 'V'] = rho * V * S * Cl
        partials['L', 'S'] = 0.5 * rho * V**2 * Cl
        partials['L', 'Cl'] = 0.5 * rho * V**2 * S


# Example usage in an optimization
if __name__ == "__main__":
    prob = om.Problem()
    
    prob.model.add_subsystem('lift', LiftForceComp(), 
                             promotes=['*'])
    
    # Add design variables for optimization
    prob.model.add_design_var('V', lower=20.0, upper=100.0)
    prob.model.add_design_var('Cl', lower=0.1, upper=2.0)
    
    # Add objective (e.g., maximize lift)
    prob.model.add_objective('L', scaler=-1.0)  # negative for maximization
    
    prob.setup()
    prob.run_model()
    
    print(f"Lift force: {prob['L'][0]:.2f} N")
    
    # Check partials
    prob.check_partials(compact_print=True)
```

**Key optimization features:**

1. **Analytical derivatives** via `compute_partials()` - much faster than finite differencing
2. **Units defined** - helps with scaling and consistency
3. **All inputs can be design variables** - flexible for different optimization scenarios
4. **Proper derivative declarations** - enables gradient-based optimization

**Alternative: Using vectorization for multiple cases:**

```python
class LiftForceCompVectorized(om.ExplicitComponent):
    """Vectorized version for analyzing multiple flight conditions"""
    
    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                            desc='Number of analysis points')
    
    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('rho', val=np.ones(nn), units='kg/m**3')
        self.add_input('V', val=np.ones(nn), units='m/s')
        self.add_input('S', val=10.0, units='m**2')
        self.add_input('Cl', val=np.ones(nn))
        
        self.add_output('L', val=np.ones(nn), units='N')
        
        # Sparse partials for efficiency
        ar = np.arange(nn)
        self.declare_partials('L', 'rho', rows=ar, cols=ar)
        self.declare_partials('L', 'V', rows=ar, cols=ar)
        self.declare_partials('L', 'S', rows=ar, cols=np.zeros(nn, dtype=int))
        self.declare_partials('L', 'Cl', rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):
        outputs['L'] = 0.5 * inputs['rho'] * inputs['V']**2 * inputs['S'] * inputs['Cl']
    
    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        Cl = inputs['Cl']
        
        partials['L', 'rho'] = 0.5 * V**2 * S * Cl
        partials['L', 'V'] = rho * V * S * Cl
        partials['L', 'S'] = np.sum(0.5 * rho * V**2 * Cl)
        partials['L', 'Cl'] = 0.5 * rho * V**2 * S
```

This component is now ready for use with any OpenMDAO optimizer (SLSQP, SNOPT, etc.)!

**Score Response 2: ___ / 8**

---

## code-gen-003 (category: code_generation_constrained)

**Prompt:** Write an ImplicitComponent that solves x^2 = a for x.

**Rubric (max 10 points):**
- [ ] uses_apply_nonlinear_for_residual (2 pts)
- [ ] residual_writes_to_residuals_not_outputs (2 pts)
- [ ] includes_jacobian_x_x_partial (3 pts)
- [ ] uses_linearize_method (1 pts)
- [ ] mentions_solver_needed_at_group_level (2 pts)

### Response 1

Here's an ImplicitComponent that solves x² = a for x:

```python
import openmdao.api as om

class QuadraticImplicit(om.ImplicitComponent):
    """
    Solves the equation x^2 = a for x using an implicit component.
    """
    
    def setup(self):
        # Input
        self.add_input('a', val=1.0, desc='Right-hand side value')
        
        # State variable (output)
        self.add_output('x', val=1.0, desc='Solution to x^2 = a')
        
        # Declare partials
        self.declare_partials('*', '*')
    
    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate residual: R = x^2 - a
        """
        a = inputs['a']
        x = outputs['x']
        
        residuals['x'] = x**2 - a
    
    def linearize(self, inputs, outputs, jacobian):
        """
        Calculate partial derivatives for the residual equation.
        """
        x = outputs['x']
        
        # dR/dx = 2*x
        jacobian['x', 'x'] = 2.0 * x
        
        # dR/da = -1
        jacobian['x', 'a'] = -1.0


# Example usage
if __name__ == '__main__':
    # Create a problem
    prob = om.Problem()
    
    # Add the implicit component
    prob.model.add_subsystem('quad', QuadraticImplicit())
    
    # Setup the problem
    prob.setup()
    
    # Set input value
    prob.set_val('quad.a', 16.0)
    
    # Set initial guess for x
    prob.set_val('quad.x', 1.0)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print(f"Input a = {prob.get_val('quad.a')[0]}")
    print(f"Solution x = {prob.get_val('quad.x')[0]}")
    print(f"Verification: x^2 = {prob.get_val('quad.x')[0]**2}")
```

**Key points:**

1. **`apply_nonlinear`**: Computes the residual R = x² - a. The solver will drive this to zero.

2. **`linearize`**: Provides derivatives:
   - `dR/dx = 2x` (partial of residual w.r.t. state variable)
   - `dR/da = -1` (partial of residual w.r.t. input)

3. **Initial guess**: Since x² = a has two solutions (±√a), the initial guess for `x` determines which solution you get. A positive initial guess gives the positive root, negative gives the negative root.

4. The nonlinear solver (default Newton) will iterate to find x such that x² - a = 0.

This component will solve for x = √a (or x = -√a depending on initial guess) when the model runs.

**Score Response 1: ___ / 10**

### Response 2

# ImplicitComponent for Square Root

Here's an ImplicitComponent that solves x² = a for x:

```python
import openmdao.api as om
import numpy as np

class SquareRootImplicit(om.ImplicitComponent):
    """
    Solves x^2 = a for x using an implicit formulation.
    Residual: R = x^2 - a = 0
    """

    def setup(self):
        # Input: the value we want to take the square root of
        self.add_input('a', val=4.0)
        
        # Output: x such that x^2 = a
        self.add_output('x', val=2.0)  # Good initial guess is important!

    def setup_partials(self):
        # Declare derivatives of the residual
        self.declare_partials('x', 'a')   # dR/da
        self.declare_partials('x', 'x')   # dR/dx (always needed for implicit!)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute the residual: R = x^2 - a
        When R = 0, we have x^2 = a (the equation is satisfied)
        """
        residuals['x'] = outputs['x']**2 - inputs['a']

    def linearize(self, inputs, outputs, jacobian):
        """
        Compute derivatives of the residual.
        """
        # dR/dx = 2*x
        jacobian['x', 'x'] = 2.0 * outputs['x']
        
        # dR/da = -1.0 (constant)
        jacobian['x', 'a'] = -1.0


# Example usage
if __name__ == '__main__':
    prob = om.Problem()
    
    # Add the implicit component
    prob.model.add_subsystem('sqrt_comp', SquareRootImplicit(),
                              promotes=['a', 'x'])
    
    # ImplicitComponents need a solver to converge the residual
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.linear_solver = om.DirectSolver()
    
    # Configure solver options
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['atol'] = 1e-10
    prob.model.nonlinear_solver.options['rtol'] = 1e-10
    
    prob.setup()
    
    # Test with different values
    test_values = [4.0, 9.0, 16.0, 25.0, 2.0]
    
    for a_val in test_values:
        prob.set_val('a', a_val)
        prob.run_model()
        
        x_result = prob.get_val('x')[0]
        expected = np.sqrt(a_val)
        error = abs(x_result - expected)
        
        print(f"a = {a_val:6.2f}  →  x = {x_result:8.6f}  "
              f"(expected: {expected:8.6f}, error: {error:.2e})")
    
    # Verify derivatives
    print("\nVerifying total derivatives with check_totals:")
    prob.set_val('a', 9.0)
    prob.run_model()
    
    prob.check_totals(of=['x'], wrt=['a'], 
                      compact_print=True, 
                      method='cs')
```

## Key Points About This Implementation

### 1. The Residual Equation
- **Goal**: Find x such that x² = a
- **Residual**: R = x² - a
- **Convergence**: When R = 0, the equation is satisfied

### 2. Why apply_nonlinear() Only Writes to residuals
```python
# CORRECT - only compute residual
residuals['x'] = outputs['x']**2 - inputs['a']

# WRONG - never modify outputs in apply_nonlinear
outputs['x'] = inputs['a']**0.5  # Don't do this!
```

### 3. Why We Need jacobian['x', 'x']
Newton's method needs the derivative of the residual with respect to the output to update the output value:
```
dR/dx = 2*x  ← This tells Newton how sensitive R is to changes in x
```

### 4. Initial Guess Matters
```python
self.add_output('x', val=2.0)  # Start with a reasonable positive value
```
If you start at x=0, the derivative dR/dx = 0 and Newton can't make progress. Always provide a good initial guess!

### 5. Alternative: With solve_nonlinear()
If you want to bypass the external solver and solve directly:

```python
def solve_nonlinear(self, inputs, outputs):
    """
    Optional: solve directly instead of using Newton solver.
    """
    outputs['x'] = inputs['a']**0.5
```

When you implement `solve_nonlinear()`, OpenMDAO uses it directly instead of iterating on the residual. This is useful when wrapping external solvers.

## Expected Output
```
a =   4.00  →  x = 2.000000  (expected: 2.000000, error: 4.44e-16)
a =   9.00  →  x = 3.000000  (expected: 3.000000, error: 0.00e+00)
a =  16.00  →  x = 4.000000  (expected: 4.000000, error: 0.00e+00)
a =  25.00  →  x = 5.000000  (expected: 5.000000, error: 0.00e+00)
a =   2.00  →  x = 1.414214  (expected: 1.414214, error: 0.00e+00)
```

This example demonstrates the essential pattern for any ImplicitComponent: define the residual in `apply_nonlinear()`, compute its derivatives in `linearize()`, and let the solver find the output value that drives the residual to zero.

**Score Response 2: ___ / 10**

---

## verification-001 (category: project_convention)

**Prompt:** I added analytic derivatives to all my components. check_partials passed. Should I optimize now?

**Rubric (max 7 points):**
- [ ] recommends_check_totals_first (3 pts)
- [ ] explains_partials_vs_totals_difference (2 pts)
- [ ] shows_verification_chain (2 pts)

### Response 1

**Not yet!** You're missing a critical step.

`check_partials()` only verifies derivatives **within each component individually**. You still need to verify **total derivatives** — the derivatives of your objective and constraints with respect to design variables **through the entire connected model** (including all coupling, connections, and solvers).

## What You Need to Do Next

```python
# After your model is set up and has run at least once
prob.run_model()

# Check total derivatives before optimizing
data = prob.check_totals(
    of=['obj', 'con1', 'con2'],   # Your objectives and constraints
    wrt=['x', 'z'],                # Your design variables
    method='cs',                   # Use complex step for accuracy
    compact_print=True
)
```

For complex step to work, you also need:
```python
prob.setup(force_alloc_complex=True)
```

## What to Look For

```
                   J_fwd          J_cs        abs err      rel err
(obj, x)       [  2.09614]   [  2.09614]   8.88e-10     4.24e-10   OK
(obj, z)       [ -3.10996    [ -3.10996    1.28e-09     4.12e-10   OK
```

- **Relative error < 1e-6**: Good — you can trust optimization
- **Relative error 1e-6 to 1e-3**: Warning — investigate further
- **Relative error > 1e-3**: Problem — you have a derivatives bug

## Why This Matters

Even if every component's partials are correct, total derivatives can still be wrong due to:
- Missing partial declarations for a connected variable
- Linear solver not converging tightly enough
- Incorrect solver configuration
- Issues in how components are connected

**The optimizer uses total derivatives, not component partials.** If total derivatives are wrong, optimization will silently fail — it may appear to run but will find wrong answers or fail to converge.

## The Complete Verification Workflow

```
1. ✓ Implement analytic partials in each component
2. ✓ Run check_partials() — passed
3. → Run check_totals() — DO THIS NOW
4.   Run optimization — only after check_totals passes
```

**Never skip step 3.** It takes a few seconds and can save you hours of debugging wrong optimization results.

**Score Response 1: ___ / 7**

### Response 2

# Not quite yet! Here's what you should do:

Even though `check_partials()` passed, you should **check your total derivatives** before optimizing:

## 1. **Run `check_totals()`**
```python
prob.run_model()
prob.check_totals(of=['objective', 'constraint1', 'constraint2'], 
                  wrt=['design_var1', 'design_var2'],
                  compact_print=True)
```

This verifies that:
- Your partial derivatives chain together correctly through the model
- There are no issues with data passing or units
- The optimizer will see the correct gradients

## 2. **What to look for:**
- **Absolute and relative errors** should be small (typically < 1e-6)
- Large errors indicate problems even if partials checked out individually

## 3. **Common issues `check_totals()` catches:**

- **Forgotten connections** between components
- **Units mismatches** that affect the chain rule
- **Implicit component solve tolerances** too loose
- **Missing or incorrect partial derivatives** in the chain
- **Problems with solvers** (linear or nonlinear)

## 4. **Only then optimize:**
```python
prob.run_driver()  # Now you're ready!
```

**Bottom line:** `check_partials()` verifies individual components, but `check_totals()` verifies the entire gradient chain that the optimizer actually uses. Both should pass before optimization.

**Score Response 2: ___ / 7**

---

## verification-002 (category: project_convention)

**Prompt:** What is the recommended workflow for adding analytic derivatives to a component?

**Rubric (max 7 points):**
- [ ] recommends_starting_with_method_fd (2 pts)
- [ ] shows_compute_partials_replacement (2 pts)
- [ ] recommends_check_partials_with_cs_method (2 pts)
- [ ] mentions_force_alloc_complex (1 pts)

### Response 1

# Recommended Workflow for Adding Analytic Derivatives

OpenMDAO recommends a systematic approach when adding analytic derivatives to a component:

## 1. **Start with Finite Difference Verification**

First, ensure your component works correctly with finite differences:
- Use `declare_partials('*', '*', method='fd')` initially
- Verify your component produces correct outputs
- Check derivatives with `check_partials()`

## 2. **Declare Your Partial Derivatives**

Specify which derivatives exist in `setup()`:

```python
def setup(self):
    # Declare partials you'll compute analytically
    self.declare_partials('output_name', 'input_name')
    
    # Can specify sparsity pattern if applicable
    self.declare_partials('y', 'x', rows=rows, cols=cols)
```

## 3. **Implement compute_partials()**

Add the `compute_partials()` method to compute derivatives:

```python
def compute_partials(self, inputs, partials):
    x = inputs['x']
    
    # Compute and store derivatives
    partials['y', 'x'] = # your derivative calculation
```

## 4. **Verify with check_partials()**

Compare your analytic derivatives against finite differences:

```python
prob.setup()
prob.run_model()
data = prob.check_partials(compact_print=True)
```

Look for small relative errors (typically < 1e-6).

## 5. **Iterate and Debug**

- If derivatives don't match, check your math and implementation
- Use `compact_print=False` for detailed output
- Check one derivative at a time if needed
- Consider using complex step (`method='cs'`) for more accurate verification

## Best Practices

- **Declare early**: Call `declare_partials()` in `setup()`, not `__init__()`
- **Use sparsity**: Specify sparse patterns when applicable to improve performance
- **Constant derivatives**: Use `val=` parameter in `declare_partials()` for constants
- **Test incrementally**: Add derivatives one at a time for complex components

**Score Response 1: ___ / 7**

### Response 2

# Recommended Workflow for Adding Analytic Derivatives

Based on the OpenMDAO intermediate skills, here's the step-by-step workflow:

## 1. Start with Finite Difference (if needed)
If you're just getting started, use `method='fd'` temporarily to verify your model logic works:

```python
def setup_partials(self):
    self.declare_partials('*', '*', method='fd')
```

## 2. Derive the Math
Work out the partial derivatives on paper or symbolically:
- For each output, determine ∂(output)/∂(input)
- Pay attention to array shapes if you have array inputs/outputs

## 3. Implement setup_partials()
Declare exactly which partials exist (remove `method='fd'`):

```python
def setup_partials(self):
    # Declare each output-input pair
    self.declare_partials('f_xy', 'x')
    self.declare_partials('f_xy', 'y')
    
    # Or declare multiple at once
    self.declare_partials('f_xy', ['x', 'y'])
```

**For constant partials**, set them directly here:
```python
def setup_partials(self):
    # If df/dx = 3.0 always
    self.declare_partials('f', 'x', val=3.0)
```

## 4. Implement compute_partials()
Set the partial values using the (output, input) key:

```python
def compute_partials(self, inputs, partials):
    x = inputs['x']
    y = inputs['y']
    
    # Set each declared partial
    partials['f_xy', 'x'] = 2.0*(x - 3.0) + y
    partials['f_xy', 'y'] = x + 2.0*(y + 4.0)
```

**For array variables**, ensure correct Jacobian shape (m×n for output size m, input size n):
```python
partials['y1', 'z'] = np.array([[2.0*inputs['z'][0], 1.0]])  # 2D array!
```

## 5. Verify with check_partials()
**This is the critical step** — always verify your analytic derivatives:

```python
prob = om.Problem()
prob.model.add_subsystem('comp', MyComponent())

# Use complex step for most accurate reference
prob.setup(force_alloc_complex=True)

prob.set_val('comp.x', 5.0)
prob.set_val('comp.y', -2.0)

prob.run_model()

# Compare analytic vs complex step
data = prob.check_partials(method='cs', compact_print=True)
```

**Look for**:
- Relative errors below **1e-6** → Good ✓
- Errors above 1e-6 → Your derivative has a bug, fix the math

## 6. Debug if Needed
If `check_partials()` shows large errors:
- Double-check your math derivation
- Verify array shapes for vector inputs/outputs
- Check for typos in variable names
- Make sure you're using the correct input values

## Complete Example: Paraboloid

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # Step 3: Declare partials
        self.declare_partials('f_xy', ['x', 'y'])

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        # Step 4: Implement derivatives
        x = inputs['x']
        y = inputs['y']
        
        # df/dx = 2*(x-3) + y
        partials['f_xy', 'x'] = 2.0*(x - 3.0) + y
        
        # df/dy = x + 2*(y+4)
        partials['f_xy', 'y'] = x + 2.0*(y + 4.0)

# Step 5: Verify!
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid())
prob.setup(force_alloc_complex=True)
prob.set_val('parab.x', 5.0)
prob.set_val('parab.y', -2.0)
prob.run_model()

prob.check_partials(method='cs', compact_print=True)
```

## Key Takeaways

1. **Always verify** — `check_partials()` is non-negotiable
2. **Use complex step** (`method='cs'`) for the reference — it's more accurate than finite difference
3. **Watch array shapes** — common source of errors
4. **Constant partials** can be declared directly in `setup_partials()`, skipping `compute_partials()`
5. **Make it a habit** — verify each component's derivatives before building the full model

This workflow ensures your analytic derivatives are correct before they cause silent optimizer failures!

**Score Response 2: ___ / 7**

---

## api-modern-001 (category: recent_api)

**Prompt:** How do I declare a design variable for optimization in OpenMDAO?

**Rubric (max 5 points):**
- [ ] uses_modern_set_val_approach (2 pts)
- [ ] does_not_unnecessarily_use_indepvarcomp (2 pts)
- [ ] shows_add_design_var_directly (1 pts)

### Response 1

# Declaring Design Variables in OpenMDAO

To declare a design variable for optimization, use `add_design_var()` on your model:

```python
import openmdao.api as om

prob = om.Problem()
model = prob.model

# Add your components
model.add_subsystem('parab', Paraboloid(), 
                    promotes=['x', 'y', 'f_xy'])

# Set up the optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# Declare design variables with bounds
prob.model.add_design_var('x', lower=-50.0, upper=50.0)
prob.model.add_design_var('y', lower=-50.0, upper=50.0)

# Also declare objective
prob.model.add_objective('f_xy')

prob.setup()
prob.run_driver()
```

## Key Points

### Required Arguments
- **First argument**: the name of the promoted variable (or 'subsystem.variable' if not promoted)
- **`lower`**: lower bound (required for most optimizers)
- **`upper`**: upper bound (required for most optimizers)

### Optional Arguments
```python
prob.model.add_design_var('x',
                           lower=0.0,
                           upper=10.0,
                           ref=10.0,      # for scaling
                           ref0=0.0,      # for scaling
                           scaler=0.1)    # alternative to ref
```

### Array Design Variables
For array variables, specify shape:

```python
import numpy as np

prob.model.add_design_var('z', 
                           lower=-10.0, 
                           upper=10.0,
                           ref=10.0)  # Applied to all elements

# Set initial value
prob.set_val('z', np.array([5.0, 2.0]))
```

### Modern OpenMDAO: No IndepVarComp Needed
You do **NOT** need `IndepVarComp` to declare design variables in modern OpenMDAO (3.x+). Just promote the input to the model level and declare it directly:

```python
# This is all you need - no IndepVarComp!
model.add_subsystem('comp', MyComponent(), promotes=['x'])
prob.model.add_design_var('x', lower=0.0, upper=10.0)
```

## Common Mistake to Avoid

**Don't forget the optimizer!** Design variables only work with an optimization driver:

```python
# WRONG - no driver assigned, add_design_var does nothing
prob.model.add_design_var('x', lower=0.0, upper=10.0)
prob.run_model()  # This just runs once with initial values

# RIGHT - assign driver before calling run_driver()
prob.driver = om.ScipyOptimizeDriver()
prob.model.add_design_var('x', lower=0.0, upper=10.0)
prob.model.add_objective('f')
prob.run_driver()  # This actually optimizes
```

Does this answer your question? Would you like to see a complete optimization example?

**Score Response 1: ___ / 5**

### Response 2

# Declaring Design Variables in OpenMDAO

To declare a design variable for optimization in OpenMDAO, you use the `add_design_var()` method on your problem object. Here's how:

## Basic Syntax

```python
prob.model.add_design_var('variable_name', lower=lb, upper=ub)
```

## Complete Example

```python
import openmdao.api as om

# Create problem and model
prob = om.Problem()
model = prob.model

# Add a component with an independent variable
model.add_subsystem('indeps', om.IndepVarComp('x', 3.0))
model.add_subsystem('paraboloid', ParaboloidComponent())

model.connect('indeps.x', 'paraboloid.x')

# Setup the optimization driver
prob.driver = om.ScipyOptimizeDriver()

# Declare design variable
prob.model.add_design_var('indeps.x', lower=-10, upper=10)

# Declare objective and constraints
prob.model.add_objective('paraboloid.f')

prob.setup()
prob.run_driver()
```

## Common Options

- **`lower`**: Lower bound (scalar or array)
- **`upper`**: Upper bound (scalar or array)
- **`ref`**: Reference value for scaling (default: 1.0)
- **`ref0`**: Zero-reference value for scaling (default: 0.0)
- **`scaler`**: Multiplier for scaling (alternative to ref)
- **`adder`**: Adder for scaling (alternative to ref0)
- **`indices`**: Specific indices of an array variable to use

## Array Design Variable Example

```python
# For array variables, specify bounds for each element
prob.model.add_design_var('indeps.x_array', 
                          lower=np.array([0, -5, -10]),
                          upper=np.array([10, 5, 0]))

# Or use indices to select specific elements
prob.model.add_design_var('indeps.x_array', 
                          lower=0, upper=10, 
                          indices=[0, 2])
```

The design variable must be an output of a component in your model, typically from an `IndepVarComp`.

**Score Response 2: ___ / 5**

---

## api-modern-002 (category: recent_api)

**Prompt:** Show me how to set input values before running my model.

**Rubric (max 4 points):**
- [ ] uses_prob_set_val (2 pts)
- [ ] shows_units_argument_optional (1 pts)
- [ ] no_unnecessary_indepvarcomp (1 pts)

### Response 1

# Setting Input Values in OpenMDAO

There are several ways to set input values in OpenMDAO before running your model:

## 1. **Using `set_val()` (Recommended)**

```python
import openmdao.api as om

# Create problem
prob = om.Problem()

# ... add systems and setup ...
prob.setup()

# Set input values using set_val()
prob.set_val('some_component.input_name', 10.0)
prob.set_val('another_comp.x', [1.0, 2.0, 3.0])  # arrays work too

# With units (OpenMDAO will convert automatically)
prob.set_val('comp.velocity', 100.0, units='km/h')

prob.run_model()
```

## 2. **Direct Assignment to `prob` (Dictionary-style)**

```python
prob.setup()

# Direct assignment after setup
prob['comp.input_name'] = 5.0
prob['comp.array_input'] = [1, 2, 3]

prob.run_model()
```

## 3. **Setting Initial Values in `setup()` Method of Components**

```python
class MyComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=5.0)  # Set default initial value
        self.add_input('y', val=np.array([1.0, 2.0, 3.0]))
```

## 4. **Using `set_input_defaults()` at Group Level**

```python
class MyGroup(om.Group):
    def setup(self):
        self.add_subsystem('comp1', MyComp())
        self.add_subsystem('comp2', AnotherComp())
        
    def configure(self):
        # Set defaults that override component defaults
        self.set_input_defaults('comp1.x', 10.0)
        self.set_input_defaults('comp2.y', [1, 2, 3], units='m')
```

## 5. **Complete Example**

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3.0)**2 + x*y + (y + 4.0)**2 - 3.0

# Build and run
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid())

prob.setup()

# Set inputs before running
prob.set_val('parab.x', 3.0)
prob.set_val('parab.y', -4.0)

prob.run_model()

print(prob.get_val('parab.f_xy'))
```

## Key Points:

- **Before `setup()`**: Use component `val` arguments or `set_input_defaults()`
- **After `setup()`**: Use `prob.set_val()` or `prob['var_name'] = value`
- **Units**: `set_val()` supports automatic unit conversion
- **Arrays**: All methods support NumPy arrays
- **Promoted names**: You can use promoted variable names if configured

The `set_val()` method is generally preferred for its clarity and unit conversion capabilities.

**Score Response 1: ___ / 4**

### Response 2

# Setting Input Values in OpenMDAO

There are several ways to set input values before running your model. Here's a complete guide:

## Modern Approach (Recommended)

Use `prob.set_val()` after `prob.setup()`:

```python
import openmdao.api as om
import numpy as np

prob = om.Problem()
model = prob.model

# Add your components
model.add_subsystem('comp', MyComponent(), promotes=['x', 'y', 'z'])

# Set up the problem
prob.setup()

# Set values AFTER setup()
prob.set_val('x', 5.0)                        # Scalar
prob.set_val('y', -2.0)
prob.set_val('z', np.array([1.0, 2.0, 3.0]))  # Array

# Now run
prob.run_model()

# Get results
output = prob.get_val('comp.output_name')
```

## Setting Values with Units

If your inputs have units, you can specify them:

```python
prob.set_val('altitude', 10000.0, units='m')
prob.set_val('velocity', 250.0, units='m/s')
```

OpenMDAO will automatically convert to the component's internal units.

## Setting Default Values in the Component

You can also set default values when declaring inputs:

```python
class MyComponent(om.ExplicitComponent):
    
    def setup(self):
        # Set default values here
        self.add_input('x', val=1.0)
        self.add_input('y', val=-2.0)
        self.add_input('z', val=np.array([5.0, 2.0]))
```

These defaults are used if you don't call `prob.set_val()` before running.

## Complete Example: Paraboloid

```python
import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

# Build and run
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(),
                         promotes_inputs=['x', 'y'])

prob.setup()

# Set input values before running
prob.set_val('x', 5.0)
prob.set_val('y', -2.0)

prob.run_model()

print(f"f_xy = {prob.get_val('parab.f_xy')[0]:.6f}")
```

## When You Have Design Variables

For optimization, you still set initial values the same way:

```python
prob = om.Problem()
model = prob.model

model.add_subsystem('parab', Paraboloid(),
                    promotes_inputs=['x', 'y'])

# Set up driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# Declare design variables
model.add_design_var('x', lower=-10.0, upper=10.0)
model.add_design_var('y', lower=-10.0, upper=10.0)
model.add_objective('parab.f_xy')

prob.setup()

# Set INITIAL values for the optimizer
prob.set_val('x', 1.0)   # Starting point
prob.set_val('y', 1.0)

# Run optimization
prob.run_driver()

print(f"Optimal x = {prob.get_val('x')[0]:.6f}")
print(f"Optimal y = {prob.get_val('y')[0]:.6f}")
```

## Key Points

✅ **Do this:**
- Always call `prob.setup()` before `prob.set_val()`
- Use promoted names if you promoted the inputs
- Use full absolute names (e.g., `'subsystem.x'`) if not promoted

❌ **Don't do this:**
- Don't try to set values before `prob.setup()` — the variables don't exist yet
- Don't confuse input names with output names

## When IndepVarComp Is NOT Needed

In modern OpenMDAO (3.x+), you do **not** need an `IndepVarComp` for simple cases. The old pattern looked like this:

```python
# OLD WAY (still works but unnecessary for simple cases)
ivc = om.IndepVarComp()
ivc.add_output('x', val=1.0)
ivc.add_output('y', val=1.0)
model.add_subsystem('ivc', ivc, promotes=['x', 'y'])
```

Just use `prob.set_val()` instead — it's simpler and clearer.

## Summary

```python
# The pattern you'll use 95% of the time:
prob.setup()          # 1. Set up the problem
prob.set_val('x', 5.0)  # 2. Set input values
prob.run_model()      # 3. Run the model
result = prob.get_val('output')  # 4. Get results
```

That's it! Do you have a specific component or model where you need help setting values?

**Score Response 2: ___ / 4**

---

## debug-001 (category: convergence_debugging)

**Prompt:** I am getting wrong results from my model. The component math looks correct. How do I debug?

**Rubric (max 7 points):**
- [ ] recommends_n2_diagram (2 pts)
- [ ] recommends_list_connections (2 pts)
- [ ] recommends_list_inputs_outputs (1 pts)
- [ ] suggests_checking_unconnected_inputs (2 pts)

### Response 1

# Debugging Strategy for Wrong Model Results

When your component math looks correct but results are wrong, follow this systematic debugging process:

## Step 1: Verify Component Derivatives First

Wrong derivatives are the #1 cause of incorrect results in optimization and coupled systems.

```python
import openmdao.api as om

prob = om.Problem()
# ... add your components ...
prob.setup(force_alloc_complex=True)  # Required for complex step

# Set reasonable input values
prob.set_val('x', 1.0)
prob.set_val('y', 2.0)

prob.run_model()

# Check derivatives for each component
data = prob.check_partials(method='cs', compact_print=True)
```

**What to look for:**
- Relative errors > 1e-6 indicate a derivative bug
- Even if you're using `method='fd'`, this will catch sign errors and missing partials
- If you have analytic derivatives, this verifies your `compute_partials()` math

## Step 2: Check if Variables Are Actually Connected

A very common mistake — thinking variables are connected when they're not.

```python
# After setup, before run_model
prob.model.list_inputs(print_arrays=True, units=True)
prob.model.list_outputs(print_arrays=True, units=True)
```

**What to look for:**
- Are promoted variables showing up with expected names?
- Are input values reasonable or still at default zeros?
- Are there duplicate names causing wrong connections?

You can also visualize connections:
```python
om.n2(prob, outfile='n2.html', show_browser=True)
```

## Step 3: Check Solvers (If You Have Coupling)

If you have coupled components (y1 depends on y2, y2 depends on y1), you need a nonlinear solver.

```python
# Check if solver is actually converging
prob.model.nonlinear_solver.options['iprint'] = 2
prob.run_model()
```

**Common issues:**
- **No solver** — coupling never converges, you get arbitrary values
- **Solver not converging** — hits maxiter, variables not at equilibrium
- **Wrong solver type** — NLBGS on strongly coupled system won't converge

**Fix:**
```python
# Add a solver if missing
prob.model.nonlinear_solver = om.NonlinearBlockGS()
prob.model.nonlinear_solver.options['maxiter'] = 100
prob.model.nonlinear_solver.options['atol'] = 1e-8
prob.model.nonlinear_solver.options['iprint'] = 2  # See convergence

# If NLBGS isn't working, try Newton
prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
prob.model.linear_solver = om.DirectSolver()
```

## Step 4: Verify Total Derivatives (If Optimizing)

If you're running an optimization and getting wrong answers:

```python
prob.setup(force_alloc_complex=True)
prob.run_model()

# Check total derivatives through the whole model
data = prob.check_totals(
    of=['obj', 'con1', 'con2'],  # Your objectives and constraints
    wrt=['x', 'z'],               # Your design variables
    method='cs',
    compact_print=True
)
```

**What to look for:**
- Relative errors > 1e-3 mean the optimizer is getting wrong gradient information
- Check if linear solver is converging tightly enough

## Step 5: Print Intermediate Values

Sometimes you just need to see what's happening inside components:

```python
class MyComponent(om.ExplicitComponent):
    
    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        
        # Debug print
        print(f"DEBUG: x={x}, y={y}")
        
        outputs['f'] = x**2 + y
        
        # Debug the output
        print(f"DEBUG: f={outputs['f']}")
```

**Pro tip:** Use Python's `pdb` debugger:
```python
def compute(self, inputs, outputs):
    import pdb; pdb.set_trace()  # Execution stops here
    # Now you can inspect inputs, outputs interactively
```

## Step 6: Check Units (If Using Physical Units)

Unit mismatches can cause factor-of-1000 type errors that "look close":

```python
# Check all variables have correct units
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
```

**Common issue:**
```python
# Component outputs in meters
self.add_output('distance', val=0.0, units='m')

# But another component expects kilometers
self.add_input('distance', val=0.0, units='km')

# OpenMDAO converts automatically, but are the units what you intended?
```

## Step 7: Isolate the Problem Component

Run components individually to verify they work in isolation:

```python
# Test one component alone
prob = om.Problem()
prob.model.add_subsystem('comp', MyComponent())
prob.setup()
prob.set_val('comp.x', 5.0)
prob.run_model()
print(prob.get_val('comp.f'))  # Does this match hand calculation?
```

## Quick Diagnostic Checklist

Run through this list:

- [ ] `check_partials()` shows errors < 1e-6 for all components
- [ ] `list_inputs()` shows variables have correct values, not zeros
- [ ] `n2()` diagram shows connections you expect
- [ ] Nonlinear solver converges (if you have coupling)
- [ ] `check_totals()` shows errors < 1e-3 (if optimizing)
- [ ] All units are correct and consistent
- [ ] Individual components give correct results in isolation

## Example: Debugging a Sellar Problem

```python
import openmdao.api as om
import numpy as np

# Assume SellarDis1, SellarDis2 defined

prob = om.Problem()
model = prob.model

model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

# Add solver for coupling
model.nonlinear_solver = om.NonlinearBlockGS()
model.nonlinear_solver.options['iprint'] = 2  # See what solver is doing

prob.setup(force_alloc_complex=True)
prob.set_val('x', 1.0)
prob.set_val('z', np.array([5.0, 2.0]))

# Step 1: Check derivatives
print("=" * 50)
print("CHECKING PARTIALS")
print("=" * 50)
prob.check_partials(method='cs', compact_print=True)

# Step 2: Run and check convergence
print("\n" + "=" * 50)
print("RUNNING MODEL")
print("=" * 50)
prob.run_model()

# Step 3: Check connections
print("\n" + "=" * 50)
print("INPUT VALUES")
print("=" * 50)
prob.model.list_inputs(print_arrays=True)

print("\n" + "=" * 50)
print("OUTPUT VALUES")
print("=" * 50)
prob.model.list_outputs(print_arrays=True)

# If optimizing, add Step 4
# prob.check_totals(method='cs', compact_print=True)
```

## Most Common Root Causes (In Order)

1. **Wrong derivatives** — check_partials() catches this
2. **Missing or non-converging solver** — coupling variables don't equilibrate
3. **Variables not connected** — check promotion or explicit connections
4. **Wrong initial guess** — especially for ImplicitComponents
5. **Unit mismatches** — check list_inputs/outputs with units=True
6. **Scaling problems** — if optimizing, variables far from order 1

**Start with check_partials() — it catches 80% of problems.**

**Score Response 1: ___ / 7**

### Response 2

# Debugging OpenMDAO Models

Here are systematic steps to debug your OpenMDAO model when the math looks correct but results are wrong:

## 1. **Check Partial Derivatives**

This is the most common issue. Use `check_partials()` to verify your derivatives:

```python
prob.setup()
prob.run_model()
prob.check_partials(compact_print=True)
```

Look for large errors in the "rel error" column. If you see errors > 1e-6, your derivatives are likely incorrect.

## 2. **Check Total Derivatives**

Verify the entire model's derivatives:

```python
prob.check_totals(of=['objective', 'constraint'], 
                  wrt=['design_var1', 'design_var2'],
                  compact_print=True)
```

## 3. **Add Debugging Outputs**

Print intermediate values in your component's `compute()` method:

```python
def compute(self, inputs, outputs):
    print(f"Inputs: {inputs}")
    outputs['y'] = 2.0 * inputs['x']
    print(f"Outputs: {outputs}")
```

## 4. **Visualize the Model**

Generate an N2 diagram to check connections:

```python
import openmdao.api as om
om.n2(prob, outfile='n2.html', show_browser=True)
```

Check for:
- Missing connections (white squares)
- Unexpected feedback loops
- Incorrect data flow

## 5. **List All Inputs/Outputs**

```python
prob.model.list_inputs(print_arrays=True, units=True)
prob.model.list_outputs(print_arrays=True, units=True)
```

Check for:
- Unconnected inputs (they'll use default values)
- Incorrect units
- NaN or unexpected values

## 6. **Common Issues to Check**

- **Units**: Are unit conversions correct?
- **Shapes**: Do array dimensions match?
- **Initial values**: Are starting guesses reasonable?
- **Solvers**: Is the solver converging? Check `prob.model.list_outputs()` for solver residuals
- **Scaling**: For optimizations, poor scaling can cause issues

## 7. **Run Model Step-by-Step**

```python
prob.setup()
prob.set_val('x', 5.0)
prob.run_model()
print(prob.get_val('y'))
```

## 8. **Check Solver Convergence**

```python
prob.set_solver_print(level=2)  # Increase verbosity
prob.run_model()
```

Would you like help with any specific aspect of debugging your model?

**Score Response 2: ___ / 7**

---



# ANSWER KEY (do not look until done scoring!)

| ID | Response 1 was | Response 2 was |
|----|---------------|----------------|
| basic-001 | No Skills | With Skills |
| basic-002 | No Skills | With Skills |
| antipattern-001 | With Skills | No Skills |
| antipattern-002 | No Skills | With Skills |
| antipattern-003 | No Skills | With Skills |
| convergence-001 | No Skills | With Skills |
| convergence-002 | With Skills | No Skills |
| code-gen-001 | No Skills | With Skills |
| code-gen-002 | With Skills | No Skills |
| code-gen-003 | No Skills | With Skills |
| verification-001 | With Skills | No Skills |
| verification-002 | No Skills | With Skills |
| api-modern-001 | With Skills | No Skills |
| api-modern-002 | No Skills | With Skills |
| debug-001 | With Skills | No Skills |
