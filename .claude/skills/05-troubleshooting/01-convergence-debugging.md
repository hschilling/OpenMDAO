# Convergence Debugging

## Purpose
Guide users through diagnosing and fixing convergence failures in OpenMDAO models.

## Key Concepts

- Convergence failures are usually due to: poor initial guesses, tight tolerances, missing/incorrect solvers, or model bugs.
- Always check solver output for clues.

## Troubleshooting Steps

1. **Check Initial Guesses**
   - Set physically reasonable starting values for all variables.
   - Use `prob.set_val()` before running.

2. **Increase maxiter**
   - Raise `maxiter` for nonlinear/linear solvers:
     ```python
     model.nonlinear_solver.options['maxiter'] = 500
     ```

3. **Relax Tolerances**
   - Loosen `atol`/`rtol` to see if the model can converge at all.

4. **Try a Different Solver**
   - Switch from NLBGS to Newton or vice versa.
   - For Newton, ensure analytic derivatives and a linear solver.

5. **Check Model Structure**
   - Use `om.n2(prob)` to visualize connections and feedback.
   - Look for missing or incorrect connections.

6. **Check for NaNs/Infs**
   - Print variable values after each run.
   - Use `list_inputs()` and `list_outputs()` to inspect.

## Anti-Patterns

- Ignoring solver warnings or errors.
- Using default initial guesses for all variables.
- Not checking for feedback loops needing a solver.

## How to Guide the User

- Ask for solver output and error messages.
- Suggest step-by-step fixes, starting with initial guesses and tolerances.
- Use N2 diagrams and list_connections for model inspection.
