#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
import openmdao.api as om

import numpy as np


# In[2]:


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
filename = "sellar_recording.sql"
recorder = om.SqliteRecorder(filename, record_viewer_data=False)
driver.recording_options['record_desvars'] = True
driver.recording_options['record_objectives'] = True
driver.recording_options['record_constraints'] = True
# driver.recording_options['live_plotting'] = dfstream
driver.recording_options['includes'] = []
driver.add_recorder(recorder)


# In[3]:


prob.run_driver()


# In[ ]:




