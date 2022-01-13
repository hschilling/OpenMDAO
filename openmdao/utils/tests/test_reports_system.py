"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys
import os

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.reports import setup_default_reports, clear_reports, set_reports_dir, \
    register_report, _reports_dir
from openmdao.visualization.n2_viewer.n2_viewer import n2

from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        import openmdao.core.problem

        self.n2_filename = 'n2.html'
        self.scaling_filename = 'driver_scaling_report.html'
        self.coloring_filename = 'jacobian_to_compute_coloring.png'

        # set things to a known initial state for all the test runs
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs

        os.environ.pop('OPENMDAO_REPORTS', None)
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        set_reports_dir('.')
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

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

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
        reports_dir = custom_dir
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')

        p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(p.is_file(),f'The scaling report file, {str(p)}, was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertTrue(p.is_file(),f'The coloring report file, {str(p)}, was not found')

    def test_report_generation_user_defined_report(self):
        from openmdao.utils.reports import register_report

        user_report_filename = 'user_report.txt'

        def user_defined_report(prob):
            with open(user_report_filename, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")

        register_report(user_defined_report, 'user defined report', 'setup', 'pre')

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

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        p = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
        self.assertTrue(p.is_file(),f'The user defined report file, {str(p)} was not found')

    def test_report_generation_various_locations(self):
        # the reports can be generated pre and post for setup, final_setup, and run_driver
        # check those all work
        from openmdao.utils.reports import register_report

        def user_defined_report(prob, filename):
            with open(filename, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                user_report_filename = f'{pre_or_post}_{method}.txt'
                register_report(user_defined_report, f'user defined report {pre_or_post} {method}', method,
                                pre_or_post, filename=user_report_filename)

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

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                user_report_filename = f'{pre_or_post}_{method}.txt'
                p = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
                self.assertTrue(p.is_file(),f'The user defined report file, {str(p)} was not found')

    def test_report_generation_multiple_problems(self):
        import openmdao.core.problem
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                p = om.Problem(name=self.prob_name)
                self._problem = p
                p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

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

        # reports_dir = '.'

        # Check existence of files for problem1
        problem1_name = p._name
        problem2_name = G.nonlinear_solver._problem._name
        for problem_name in [problem1_name, problem2_name]:
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}_reports')
            p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
            p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertFalse(p.is_file(),f'The scaling report file, {str(p)}, was found but should not have')
            p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
            self.assertFalse(p.is_file(),f'The coloring report file, {str(p)}, was found but should not have')

    def test_report_generation_multiple_problems_report_specific_problem(self):
        # test the ability to register a report with a specific Problem name rather
        #   than have the report run for all Problems
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                p = om.Problem(name=self.prob_name)
                self._problem = p
                p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

                return super().solve()

        # to simplify things, just do n2. There are issues with coloring and scaling reports
        clear_reports()
        register_report(n2, 'create n2', 'final_setup', 'post', probname='problem2',
                        show_browser=False)

        # Initially use the default names
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()  # need to do run_model in this test so sub problem is created

        probname = p._name
        subprobname = G.nonlinear_solver._problem._name
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{subprobname}_reports')

        p = pathlib.Path(problem_reports_dir).joinpath('n2.html')
        self.assertTrue(p.is_file(),f'The user defined report file, {str(p)} was not found')

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{probname}_reports')
        p = pathlib.Path(problem_reports_dir).joinpath('n2.html')
        self.assertFalse(p.is_file(), f'The user defined report file, {str(p)} was found but should not have')

    def test_report_generation_test_TESTFLO_RUNNING(self):
        os.environ['TESTFLO_RUNNING'] = 'true'

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

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        p = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(p.is_file(),f'The N2 report file, {str(p)} was found but should not have')
        p = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(p.is_file(),f'The scaling report file, {str(p)}, was found but should not have')
        p = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertFalse(p.is_file(),f'The coloring report file, {str(p)}, was not found')



if __name__ == '__main__':
    unittest.main()
