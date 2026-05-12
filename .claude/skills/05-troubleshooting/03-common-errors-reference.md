# Common Errors Reference

## Purpose
Provide a quick reference for common OpenMDAO error messages and their fixes.

## Error Table

| Error Message | Likely Cause | Fix |
|---------------|--------------|-----|
| `KeyError: 'var'` | Typo in variable name or wrong path | Check spelling, check promotes/connection |
| `RuntimeError: ... has no connected output` | Input not connected | Add `connect()` or check `promotes` |
| `SetupError: ... multiple outputs ...` | Two outputs promoted with same name | Rename or use explicit `connect()` |
| `SolverError: ... failed to converge` | Nonlinear solver did not reach tolerance | See convergence debugging skill |
| `ValueError: shape mismatch` | Connected variables have different shapes | Check `val=` shapes in setup |
| `TypeError: ... cannot be interpreted as an integer` | Passing float where int expected | Check variable types in setup/compute |
| `AttributeError: 'NoneType' object has no attribute ...` | Uninitialized variable or missing return | Check all outputs are set in compute |
| `ImportError: No module named 'mpi4py'` | Running in parallel without mpi4py | Install mpi4py or run serially |

## How to Guide the User

- Ask for the full error message and traceback.
- Suggest fixes based on the table above.
- If error is not listed, recommend searching OpenMDAO docs or forums.
