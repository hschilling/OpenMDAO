# Groups and Hierarchy

## Purpose
Teach the user how to organize components into groups, why hierarchy matters, and how variable naming works across levels.

## Key Concepts

### What Is a Group?
A Group is a container for components and other groups. It creates hierarchy and organization in your model. Every OpenMDAO Problem has a top-level group called `model`.

```python
prob = om.Problem()

# prob.model is the top-level group
prob.model.add_subsystem('comp1', MyComponent())
```

### Creating Custom Groups
When your model has logical subsystems, encapsulate them in a Group subclass:

```python
class FlightDynamics(om.Group):

    def setup(self):
        self.add_subsystem('aero', AeroComponent())
        self.add_subsystem('propulsion', PropulsionComponent())
        self.add_subsystem('eom', EquationsOfMotion())

        # Internal connections
        self.connect('aero.drag', 'eom.drag')
        self.connect('propulsion.thrust', 'eom.thrust')
```

Then use it like a component:
```python
prob = om.Problem()
prob.model.add_subsystem('flight', FlightDynamics())
```

### Variable Naming and Paths
Variables are addressed using dot-separated paths from the top level:
- `flight.aero.drag` — the `drag` output inside the `aero` component inside the `flight` group
- When a variable is promoted, it can be accessed at the group level: if `aero` promotes `drag`, it becomes `flight.drag`

### Promotes at the Group Level
Groups can also promote variables to expose them to their parent:

```python
prob.model.add_subsystem('flight', FlightDynamics(),
                          promotes_inputs=['altitude', 'mach'])
```

This makes `altitude` and `mach` accessible at the model level as `altitude` and `mach` rather than `flight.altitude` and `flight.mach`.

### When to Create a Group
- When you have 3+ components that logically belong together
- When you want to reuse a subsystem in multiple models
- When you want to apply a solver to a specific subset of components (solvers are assigned to groups)
- When your model is getting hard to read as a flat list of components

## Anti-Patterns to Watch For

### Flat Models That Should Be Grouped
**Wrong instinct:** "I'll just add all 20 components to prob.model."
**Guide the user to:** Group related components. It makes the model readable, reusable, and easier to debug. It also allows you to assign solvers to specific coupled subsystems.

### Over-Promoting Through Multiple Levels
**Wrong instinct:** Using `promotes=['*']` at every level of hierarchy.
**Why it's wrong:** Variable names can collide unexpectedly across groups. The hierarchy exists partly to provide namespacing.
**Guide the user to:** Only promote variables that genuinely need to be visible to the parent level. Keep internal wiring internal using `connect()` within the group.

## How to Guide the User
- If the user has more than 3-4 components, suggest grouping them
- If the user is confused about variable paths, draw out the hierarchy and show how dots map to levels
- Remind users that solvers are assigned to groups — this becomes important for convergence
- If the user asks "where do I put my solver," the answer is: on the group that contains the coupled components
