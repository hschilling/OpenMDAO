# External Code Wrapping Recipe

## Purpose
Show how to wrap external codes (Fortran, C, MATLAB, etc.) as OpenMDAO components.

## Key Concepts

- Use Python subprocess, f2py, ctypes, or other wrappers
- Inputs/outputs mapped to OpenMDAO variables
- Error handling and data conversion

## Pattern

### Wrapping a Fortran/C code with subprocess

```python
import openmdao.api as om
import subprocess

class ExternalComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('input1', val=0.0)
        self.add_output('output1', val=0.0)

    def setup_partials(self):
        self.declare_partials('output1', 'input1', method='fd')

    def compute(self, inputs, outputs):
        # Example: call external executable
        cmd = ['./my_external_code', str(inputs['input1'])]
        result = subprocess.run(cmd, capture_output=True, text=True)
        outputs['output1'] = float(result.stdout.strip())
```

### Wrapping MATLAB with pymatlab or matlab.engine

```python
import openmdao.api as om
import matlab.engine

class MatlabComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=0.0)

    def setup_partials(self):
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        eng = matlab.engine.start_matlab()
        y = eng.my_matlab_function(inputs['x'])
        outputs['y'] = y
        eng.quit()
```

## Anti-Patterns

- Not handling errors from external code.
- Not converting data types properly.

## How to Guide the User

- Always validate input/output mapping.
- Use method='fd' for partials unless analytic derivatives are available.
- Test external code integration in isolation first.
