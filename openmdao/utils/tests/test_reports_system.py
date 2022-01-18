"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys
import os
from io import StringIO

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
import openmdao.core.problem
from openmdao.utils.reports import setup_default_reports, clear_reports, set_reports_dir, \
    register_report, list_reports, _reports_dir, clear_reports_run
from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.n2_filename = 'n2.html'
        self.scaling_filename = 'driver_scaling_report.html'
        self.coloring_filename = 'jacobian_to_compute_coloring.png'

        # set things to a known initial state for all the test runs
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs

        os.environ.pop('OPENMDAO_REPORTS', None)
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        # We need to remove this for these tests to run. The reports code
        #   checks to see if TESTFLO_RUNNING is set and will not do anything if it is set
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        clear_reports_run()
        set_reports_dir('.')
        setup_default_reports()

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def setup_and_run_simple_problem(self):
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

        return prob

    def setup_and_run_model_with_subproblem(self):
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                subprob = om.Problem(name=self.prob_name)
                self._problem = subprob
                subprob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                subprob.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                subprob.model.connect('indep.x', 'comp.x')
                subprob.setup()
                subprob.run_model()

                return super().solve()

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = prob.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        prob.model.connect('indep.x', 'G.comp.x')
        prob.setup()
        prob.run_model()  # need to do run_model in this test so sub problem is created

        probname = prob._name
        subprobname = G.nonlinear_solver._problem._name
        return probname, subprobname

    def test_report_generation_basic(self):
        prob = self.setup_and_run_simple_problem()

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertTrue(path.is_file(), f'The coloring report file, {str(path)}, was not found')

    def test_report_generation_list_reports(self):
        self.setup_and_run_simple_problem()

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            list_reports()
        finally:
            sys.stdout = stdout

        output = strout.getvalue()

        self.assertTrue('N2 diagram' in output)
        self.assertTrue('Driver scaling report' in output)
        self.assertTrue('Coloring report' in output)

    def test_report_generation_no_reports(self):
        os.environ['OPENMDAO_REPORTS'] = 'false'

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertFalse(path.is_file(),
                         f'The coloring report file, {str(path)}, was found but should not have')

    def test_report_generation_set_reports_dir(self):
        custom_dir = 'custom_reports_dir'
        os.environ['OPENMDAO_REPORTS_DIR'] = custom_dir

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        reports_dir = custom_dir
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertTrue(path.is_file(), f'The coloring report file, {str(path)}, was not found')

    def test_report_generation_user_defined_report(self):
        user_report_filename = 'user_report.txt'

        def user_defined_report(problem):
            with open(user_report_filename, "w") as f:
                f.write(f"Do some reporting on the Problem, {problem._name}\n")

        register_report(user_defined_report, 'user defined report', 'setup', 'pre')

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
        self.assertTrue(path.is_file(), f'The user defined report file, {str(path)} was not found')

    def test_report_generation_various_locations(self):
        # the reports can be generated pre and post for setup, final_setup, and run_driver
        # check those all work

        def user_defined_report(problem, filename):
            with open(filename, "w") as f:
                f.write(f"Do some reporting on the Problem, {problem._name}\n")

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                user_report_filename = f'{pre_or_post}_{method}.txt'
                register_report(user_defined_report, f'user defined report {pre_or_post} {method}',
                                method, pre_or_post, filename=user_report_filename)

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                user_report_filename = f'{pre_or_post}_{method}.txt'
                path = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
                self.assertTrue(path.is_file(),
                                f'The user defined report file, {str(path)} was not found')

    def test_report_generation_multiple_problems(self):
        probname, subprobname = self.setup_and_run_model_with_subproblem()

        for problem_name in [probname, subprobname]:
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}_reports')
            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(),
                            f'N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertFalse(path.is_file(),
                             f'Scaling report file, {str(path)}, was found but should not have')
            path = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
            self.assertFalse(path.is_file(),
                             f'Coloring report file, {str(path)}, was found but should not have')

    def test_report_generation_multiple_problems_report_specific_problem(self):
        # test the ability to register a report with a specific Problem name rather
        #   than have the report run for all Problems

        # to simplify things, just do n2.
        clear_reports()
        register_report(n2, 'create n2', 'final_setup', 'post', probname='problem2',
                        show_browser=False)

        probname, subprobname = self.setup_and_run_model_with_subproblem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{subprobname}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath('n2.html')
        # for the subproblem Problem named problem2, there should be a report but not for problem1
        self.assertTrue(path.is_file(), f'The n2 report file, {str(path)} was not found')

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{probname}_reports')
        path = pathlib.Path(problem_reports_dir).joinpath('n2.html')
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')

    def test_report_generation_test_TESTFLO_RUNNING(self):
        os.environ['TESTFLO_RUNNING'] = 'true'

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.coloring_filename)
        self.assertFalse(path.is_file(),
                         f'The coloring report file, {str(path)}, was found but should not have')


if __name__ == '__main__':
    unittest.main()
