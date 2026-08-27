import numpy as np

import openmdao.api as om

prob = om.Problem()

sin_mm = om.MetaModelUnStructuredComp()
sin_mm.add_input('x', 2.1)
sin_mm.add_output('f_x', 0., surrogate=om.ResponseSurface())

prob.model.add_subsystem('sin_mm', sin_mm)

prob.setup()

# train the surrogate and check predicted value
sin_mm.options['train_x'] = np.linspace(0, 3.14, 20)
sin_mm.options['train_f_x'] = .5*np.sin(sin_mm.options['train_x'])

prob.set_val('sin_mm.x', 2.1)

from time import perf_counter

start_time = perf_counter()
prob.run_model()
end_time = perf_counter()

print(f"fit time = {end_time-start_time}")

print(prob.get_val('sin_mm.f_x'))
