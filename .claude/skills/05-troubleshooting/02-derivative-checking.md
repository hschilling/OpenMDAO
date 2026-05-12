# Derivative Checking

## Purpose
Help users debug and verify partial and total derivatives in OpenMDAO models.

## Key Concepts

- Correct derivatives are essential for optimization and Newton solvers.
- Use `check_partials()` for component-level, `check_totals()` for model-level.

## Troubleshooting Steps

1. **Run check_partials()**
   - After `prob.run_model()`, call:
     ```python
     prob.check_partials(compact_print=True)
     ```
   - Look for large relative errors (>1e-6).

2. **Run check_totals()**
   - After `prob.run_model()`, call:
     ```python
     prob.check_totals(compact_print=True)
     ```
   - Use `method='cs'` (complex step) for best accuracy.

3. **Interpret Output**
   - Large errors: likely a bug in `compute_partials()` or missing partials.
   - Small errors: likely OK.

4. **Common Fixes**
   - Check Jacobian shapes for array variables.
   - Remove `method='fd'` if using analytic derivatives.
   - Ensure all declared partials are set in `compute_partials()`.

## Anti-Patterns

- Skipping derivative checks before optimization.
- Trusting finite difference as the reference (use complex step if possible).
- Not checking all design variable/objective/constraint pairs.

## How to Guide the User

- Always run `check_partials()` and `check_totals()` before trusting optimization results.
- If errors are large, debug one component at a time.
- Use analytic derivatives for performance and accuracy.
