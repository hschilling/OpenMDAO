# OpenMDAO Intermediate Skills

These skills are for users who have completed the beginner skills and can build, connect, and run basic OpenMDAO models. They are organized in a logical progression — each skill builds on previous ones.

## Prerequisites
Complete the beginner skills in 01-beginner-tasks/ before using these.

## Files
1. `01-analytic-derivatives.md` — Writing compute_partials() to replace finite difference
2. `02-implicit-components.md` — ImplicitComponent, residuals, linearize()
3. `03-nonlinear-solvers.md` — NLBGS vs Newton, solver configuration
4. `04-linear-solvers.md` — DirectSolver vs iterative, total derivative computation
5. `05-execcomp.md` — ExecComp shortcut for simple expressions
6. `06-indepvarcomp.md` — IndepVarComp, when it is and is not needed
7. `07-design-of-experiments.md` — DOEDriver, sampling strategies, reading results
8. `08-case-recording.md` — SqliteRecorder, CaseReader, tracking optimization history
9. `09-scaling.md` — Variable scaling, ref/ref0, diagnosing scaling problems
10. `10-check-totals.md` — Verifying total derivatives before optimization

## Backbone Example
The Sellar problem is used throughout these skills as the canonical coupled MDO example:
- SellarDis1: y1 = z[0]**2 + z[1] + x - 0.2*y2
- SellarDis2: y2 = y1**0.5 + z[0] + z[1]
- Objective: obj = x**2 + z[1] + y1 + exp(-y2)
- Constraints: con1 = 3.16 - y1 <= 0, con2 = y2 - 24.0 <= 0

## Key Anti-Patterns Covered
- Wrong Jacobian shape for array inputs (01)
- Confusing residuals with outputs in ImplicitComponent (02)
- Assigning solver to wrong group level (03)
- Using DirectSolver on large models (04)
- Overusing ExecComp for complex logic (05)
- Adding IndepVarComp unnecessarily in modern OpenMDAO (06)
- Confusing DOE with optimization (07)
- Forgetting prob.cleanup() after recording (08)
- Skipping scaling entirely (09)
- Skipping check_totals() before optimization (10)
