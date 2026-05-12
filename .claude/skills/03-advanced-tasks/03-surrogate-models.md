# Surrogate Models

## Purpose
Guide users in using surrogate models/metamodels for fast approximations and optimization.

## Key Concepts

### Metamodel Components
- Use `om.MetaModelUnStructuredComp` for surrogate modeling.
- Train with data, then use for prediction.

```python
import openmdao.api as om
import numpy as np

mm = om.MetaModelUnStructuredComp()
mm.add_input('x', 0.0)
mm.add_input('y', 0.0)
mm.add_output('f_xy', 0.0)

# Attach a surrogate (e.g., Linear, RBF, KNN)
from openmdao.surrogate_models import RBF
mm.add_surrogate('f_xy', RBF())

# Provide training data
mm.add_training_data('x', np.array([1.0, 2.0, 3.0]))
mm.add_training_data('y', np.array([4.0, 5.0, 6.0]))
mm.add_training_data('f_xy', np.array([7.0, 8.0, 9.0]))
```

### Using Surrogates in Optimization
- Replace expensive components with metamodels for fast optimization.
- Use `MetaModel` for design space exploration.

## Anti-Patterns to Watch For

- Using surrogates without enough training data: poor predictions.
- Not validating surrogate accuracy before optimization.

## How to Guide the User

- Always validate surrogate with test data.
- Use surrogates for rapid prototyping, then switch to full models for final runs.
