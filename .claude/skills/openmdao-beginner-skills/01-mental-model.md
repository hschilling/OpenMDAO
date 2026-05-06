# OpenMDAO Mental Model

## Purpose
Establish the foundational mental model for how OpenMDAO works so that all subsequent guidance builds on correct intuition.

## Key Concepts

### What OpenMDAO Is
OpenMDAO is a framework for building and solving multidisciplinary models. It is not a solver itself — it is the scaffolding that organizes your engineering calculations and connects them to solvers, optimizers, and other drivers.

### The Core Metaphor: A Dataflow Graph, Not a Script
Traditional engineering scripts are procedural: call function A, pass the result to function B, etc. OpenMDAO is different. You describe the **structure** of your model — what the components are, what their inputs and outputs are, and how they connect — and OpenMDAO figures out the execution order and manages the data flow.

Think of it as building a circuit board:
- You place chips (components) on the board
- You wire them together (connections)
- You plug it in and let the system run (driver + solvers)

You do NOT manually pass data between components. OpenMDAO does that for you.

### The Five Building Blocks

1. **Component** — A single unit of calculation. Inputs go in, outputs come out. Each component should do ONE thing well. There are two types:
   - `ExplicitComponent`: outputs are computed directly from inputs (y = f(x))
   - `ImplicitComponent`: outputs satisfy a residual equation (r(x, y) = 0)

2. **Group** — A container that holds components (and/or other groups). Groups create hierarchy and organization. Every OpenMDAO model has at least one top-level group called `model`.

3. **Connection** — A wire between one component's output and another component's input. Data flows along connections. There are two ways to make connections:
   - `connect()`: explicitly wire output A to input B by name
   - `promotes`: expose a variable up to the group level so that variables with the same promoted name are automatically connected

4. **Driver** — The "why" of your model. It decides the purpose of running:
   - `RunOnce`: just execute the model once
   - `ScipyOptimizeDriver`: run an optimization
   - `DOEDriver`: run a design of experiments

5. **Solver** — The "how" of convergence. When components form a coupled loop (A depends on B, B depends on A), solvers iterate until the system converges. Two types:
   - `NonlinearSolver`: converges the actual values
   - `LinearSolver`: converges the derivatives (used behind the scenes for optimization)

### How Execution Works
1. You build the model: create components, add them to groups, make connections
2. You call `setup()`: OpenMDAO analyzes the structure, determines execution order, allocates memory
3. You call `run_model()` (or `run_driver()` for optimization): OpenMDAO executes components in the correct order, passing data along connections, using solvers to converge any coupled loops
4. You read the results from the model's variables

### The Variable System
- Every variable has a **name**, a **value**, and optionally **units**
- OpenMDAO handles unit conversions automatically when connected variables have different but compatible units
- Variables are either **inputs** (consumed by a component) or **outputs** (produced by a component)
- An input on one component is connected to an output on another component — never input-to-input or output-to-output

## Anti-Patterns to Watch For

### Monolithic Components
**Wrong instinct:** "I'll put my entire analysis in one big component."
**Why it's wrong:** You lose the ability to swap parts, add optimization, or debug individual pieces. OpenMDAO's power comes from decomposition.
**Guide the user to:** Break calculations into logical pieces. Each component should represent one discipline, one equation set, or one physical transformation. If a component's `compute()` method is getting long, it probably should be split.

### Thinking Procedurally
**Wrong instinct:** "I need to call component A first, then pass its output to component B."
**Why it's wrong:** OpenMDAO determines execution order automatically from the connection graph. You describe structure, not sequence.
**Guide the user to:** Focus on declaring inputs, outputs, and connections. Trust OpenMDAO to figure out the order.

## How to Guide the User
- If the user is brand new, explain the circuit board metaphor before showing any code
- If the user jumps straight to code, gently check: "Are you comfortable with how OpenMDAO organizes models into components, groups, and connections?" If not, explain first
- Use the five building blocks (Component, Group, Connection, Driver, Solver) as a consistent vocabulary throughout all conversations
- When a user's question maps to one of the building blocks, name it explicitly: "What you're describing is a Group" or "That's a connection between two components"
