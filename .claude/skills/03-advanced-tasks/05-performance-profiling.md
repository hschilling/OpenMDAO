# Performance Profiling

## Purpose
Show how to profile, time, and optimize OpenMDAO models for speed and efficiency.

## Key Concepts

### Timing Tools
- Use Python’s `time` module or `cProfile` for timing.
- OpenMDAO provides built-in profiling via `recorders`.

```python
import time
start = time.time()
prob.run_model()
print("Elapsed:", time.time() - start)
```

### Profiling with Recorder
- Use `SqliteRecorder` to record timings and variable values.

```python
recorder = om.SqliteRecorder('perf.sql')
prob.driver.add_recorder(recorder)
prob.setup()
prob.run_driver()
```

### Analyzing Bottlenecks
- Use `CaseReader` to analyze recorded data.
- Identify slow components and optimize their code.

### Optimizing Model Structure
- Decompose large components.
- Use analytic derivatives for speed.
- Avoid unnecessary data transfers.

## Anti-Patterns to Watch For

- Profiling only the full model: miss slow subsystems.
- Ignoring solver timings: solvers often dominate runtime.

## How to Guide the User

- Profile both at the model and subsystem level.
- Use analytic derivatives and sparse Jacobians.
- Optimize slow components first.
