from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
import openmdao.api as om

import numpy as np

import pandas as pd

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


driver = prob.driver
filename = "sellar_recording.sql"
recorder = om.SqliteRecorder(filename, record_viewer_data=False)
driver.recording_options['record_desvars'] = True
driver.recording_options['record_objectives'] = True
driver.recording_options['record_constraints'] = True
driver.recording_options['includes'] = []
driver.add_recorder(recorder)



failed = prob.run_driver()

assert_near_equal(prob['z'][0], 1.9776, 1e-3)
assert_near_equal(prob['z'][1], 0.0, 1e-3)
assert_near_equal(prob['x'], 0.0, 1e-3)

cr = om.CaseReader(filename)
driver_cases = cr.list_cases('driver', out_stream=None)

df = pd.DataFrame()


for i, iter_coord in enumerate(driver_cases):
    case = cr.get_case(iter_coord)
    for key, value in case.outputs.items():
        if key not in df.columns:
            df[key] = None  # None represents the initial values in the column
        df.loc[i, key] = np.linalg.norm(value)

print(df.head())


# https://towardsdatascience.com/3-ways-to-build-a-panel-visualization-dashboard-6e14148f529d


import pandas as pd
import numpy as np
import panel as pn
pn.extension('tabulator')

import hvplot.pandas

pd.options.plotting.backend = 'holoviews'


idf = df.interactive()

driver_pipeline = (
    idf
    .to_frame()
    .reset_index()
    .reset_index(drop=True)
)

co2_plot = driver_pipeline.hvplot(x = 'iteration', y='z',line_width=2, title="z output")
co2_plot




