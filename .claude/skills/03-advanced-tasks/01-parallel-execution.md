# Parallel Execution

## Purpose
Guide users in running OpenMDAO models in parallel using MPI, distributed components, and parallel drivers.

## Key Concepts

### MPI and Parallel Models
- OpenMDAO supports parallel execution via MPI (Message Passing Interface).
- To run in parallel, install mpi4py and launch your script with `mpirun` or `mpiexec`.

```bash
mpirun -np 4 python my_model.py
```

### Distributed Components
- Use `distributed=True` in `add_subsystem()` to distribute a component across processes.
- Each process gets a subset of the data.

```python
model.add_subsystem('dist_comp', MyDistributedComp(), distributed=True)
```

### Parallel Drivers
- `pyOptSparseDriver` and `DOEDriver` can run cases in parallel.
- Set `run_parallel=True` in the driver.

```python
prob.driver = om.DOEDriver(generator)
prob.driver.options['run_parallel'] = True
```

### Parallel Groups
- Use `om.ParallelGroup` to run subsystems in parallel.

```python
class MyParallelGroup(om.ParallelGroup):
    def setup(self):
        self.add_subsystem('comp1', Comp1(), promotes=['*'])
        self.add_subsystem('comp2', Comp2(), promotes=['*'])
```

## Anti-Patterns to Watch For

- Running with `mpirun` but not using distributed components: no speedup.
- Using distributed components without proper MPI setup: errors or incorrect results.
- Not checking for race conditions or data dependencies.

## How to Guide the User

- Always test serial first, then scale up.
- Use `mpirun` for parallel runs.
- For distributed arrays, use `self.comm` and `self.rank` in your component.
- Profile parallel performance with timing tools.
