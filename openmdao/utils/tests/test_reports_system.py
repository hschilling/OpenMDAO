"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.utils.testing_utils import use_tempdirs

import openmdao

# @use_tempdirs
class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        from openmdao.core.problem import setup_default_reports
        self.n2_filename = 'n2.html'
        self.scaling_filename = 'driver_scaling_report.html'
        self.coloring_filename = 'jacobian_to_compute_coloring.png'
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs

        import os
        # del os.environ['OPENMDAO_REPORTS']
        os.environ.pop('OPENMDAO_REPORTS', None)
        setup_default_reports()

    def test_report_generation_basic(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # get the report dir path
        import inspect
        script_path = inspect.stack()[-1][1]
        script_name = pathlib.Path(script_path).stem
        # reports_dir = f'{script_name}_reports'
        reports_dir = '.'

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')

        p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(p.is_file(),f'The scaling report file, {str(p)}, was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertTrue(p.is_file(),f'The coloring report file, {str(p)}, was not found')

    def test_report_generation_list_reports(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.ScipyOptimizeDriver()

        from openmdao.utils.reports import list_reports

        from io import StringIO
        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            list_reports()
        finally:
            sys.stdout = stdout

        output = strout.getvalue()

        self.assertTrue('n2' in output)
        self.assertTrue('view_driver_scaling' in output)
        self.assertTrue('coloring_reporting' in output)


    def test_report_generation_no_reports(self):
        import os
        os.environ['OPENMDAO_REPORTS'] = 'false'
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # See if the report files exist and if they have the right names
        # script_name = pathlib.Path(sys.argv[-1]).stem
        # reports_dir = f'{script_name}_reports'
        reports_dir = '.'
        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')

        # n2_filename = f'{prob._name}_N2.html'
        # scaling_filename = f'{prob._name}_driver_scaling.html'
        # n2_filename = 'n2.html'
        # scaling_filename = 'driver_scaling_report.html'

        p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(p.is_file(),f'The N2 report file, {str(p)} was found but should not have')
        p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(p.is_file(),f'The scaling report file, {str(p)}, was found but should not have')
        p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertFalse(p.is_file(),f'The coloring report file, {str(p)}, was not found')

    def test_report_generation_set_reports_dir(self):
        import os

        custom_dir = 'custom_reports_dir'
        os.environ['OPENMDAO_REPORTS_DIR'] = custom_dir
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # See if the report files exist and if they have the right names
        # script_name = pathlib.Path(sys.argv[-1]).stem
        reports_dir = custom_dir
        # n2_filename = 'n2.html'
        # scaling_filename = 'driver_scaling_report.html'
        # coloring_filename = 'jacobian_to_compute_coloring.png'

        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')


        p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(p.is_file(),f'The scaling report file, {str(p)}, was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertTrue(p.is_file(),f'The coloring report file, {str(p)}, was not found')

    def test_report_generation_multiple_problems(self):
        import openmdao.core.problem
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                # create a simple subproblem and run it to test for global solver_info bug
                p = om.Problem(name=self.prob_name)
                self._problem = p
                p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

                print(p.msginfo)

                return super().solve()

        # Initially use the default names
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()  # need to do run_model in this test so sub problem is created





if __name__ == '__main__':
    unittest.main()
