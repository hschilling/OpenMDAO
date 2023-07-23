#!/usr/bin/env python
# coding: utf-8

from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
import openmdao.api as om

from openmdao.recorders.plotting_recorder import PlottingRecorder

import numpy as np

prob = om.Problem()
model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                              linear_solver=om.ScipyKrylov)

prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', upper=0.0)
model.add_constraint('con2', upper=0.0)

prob.setup(check=False, mode='rev')

# point_source = streamz.Stream()

driver = prob.driver
recorder = PlottingRecorder()
driver.add_recorder(recorder)

prob.set_val('x', 3.0)
prob.set_val('z', np.array([2.0, 2.0]))
# prob.final_setup()
prob.run_driver()

