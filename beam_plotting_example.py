import openmdao.api as om
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.mpi import MPI
from openmdao.recorders.plotting_recorder import PlottingRecorder


E = 1.
L = 1.
b = 0.1
volume = 0.01

num_elements = 50

prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()
recorder = PlottingRecorder()
prob.driver.add_recorder(recorder)

prob.run_driver()
