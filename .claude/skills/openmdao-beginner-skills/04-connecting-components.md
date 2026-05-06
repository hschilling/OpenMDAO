# Connecting Components

## Purpose
Teach the user the two ways to connect components and when to use each. This addresses one of the most common stumbling blocks for new users.

## Key Concepts

### Why Connections Matter
Components do not call each other. They declare inputs and outputs, and OpenMDAO passes data between them via connections. If two components are not connected, data does NOT flow between them, even if variable names match.

### Method 1: Explicit connect()
You tell OpenMDAO exactly which output feeds which input.

```python
prob = om.Problem()
model = prob.model

model.add_subsystem('comp1', Paraboloid())
model.add_subsystem('comp2', AnotherComponent())

# Connect comp1's output 'f_xy' to comp2's input 'incoming_value'
model.connect('comp1.f_xy', 'comp2.incoming_value')
```

- Uses full path names: `'component_name.variable_name'`
- Explicit and unambiguous — you can always see what is wired to what
- Good for clarity when variable names differ between components

### Method 2: promotes
You expose variables up to the group level. Variables with the same promoted name are automatically connected.

```python
prob = om.Problem()
model = prob.model

model.add_subsystem('comp1', Paraboloid(),
                     promotes_outputs=['f_xy'])
model.add_subsystem('comp2', AnotherComponent(),
                     promotes_inputs=['f_xy'])
```

- `comp1` promotes its output `f_xy` up to the model level
- `comp2` promotes its input `f_xy` up to the model level
- Because they share the promoted name `f_xy`, they are automatically connected
- You can also use `promotes=['*']` to promote everything — but be careful

### Renaming During Promotion
You can promote a variable under a different name:
```python
model.add_subsystem('comp1', Paraboloid(),
                     promotes_outputs=[('f_xy', 'shared_value')])
model.add_subsystem('comp2', AnotherComponent(),
                     promotes_inputs=[('incoming_value', 'shared_value')])
```
Both variables are now known as `shared_value` at the model level and are connected.

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Variable names match naturally | `promotes` is clean and readable |
| Variable names differ | `connect()` or rename during promotion |
| Small model, few components | Either works |
| Large model, many components | Be deliberate — prefer `connect()` or carefully scoped promotes |
| Debugging connection issues | `connect()` is easier to trace |

## Anti-Patterns to Watch For

### Silent Variable Collisions with promotes
**This is a critical beginner mistake.**

**Wrong instinct:** "I'll just use `promotes=['*']` on everything."
**What goes wrong:** If two components both promote an output with the same name, OpenMDAO will raise an error. But if one promotes an output and an unrelated component promotes an input with the same name, they silently connect — even if you didn't intend them to.

```python
# DANGEROUS: unintended silent connection
model.add_subsystem('aero', AeroComp(),
                     promotes=['*'])  # has output 'temperature'
model.add_subsystem('thermal', ThermalComp(),
                     promotes=['*'])  # has input 'temperature'
# These are now connected! Was that intended?
```

**Guide the user to:**
- Be explicit about what you promote: list variable names instead of using `'*'`
- Use `promotes_inputs` and `promotes_outputs` separately instead of `promotes`
- When in doubt, use `connect()` — it is never ambiguous
- Use `model.list_connections()` to verify connections are what you expect

### Forgetting to Connect
**Wrong instinct:** "I named them the same thing, so they must be connected."
**Why it's wrong:** Without `connect()` or `promotes`, matching names mean nothing. The input will just use its default value.
**Guide the user to:** Always verify connections with `model.list_connections()` after setup.

## How to Guide the User
- For first-time users building a simple model: start with `connect()` because it is the most explicit and debuggable
- Once they understand connections, introduce `promotes` as a convenience
- If the user has a connection bug, immediately suggest `prob.model.list_connections()` to inspect
- If the user uses `promotes=['*']`, warn them about silent collisions and suggest being explicit
- Always ask: "Are all your inputs receiving data from somewhere? Any unconnected inputs will just use their default value."
