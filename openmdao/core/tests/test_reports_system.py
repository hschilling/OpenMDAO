"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.utils.testing_utils import use_tempdirs

# @use_tempdirs
class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        from openmdao.core.problem import setup_default_reports
        setup_default_reports()

    def test_basic_report_generation(self):
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
        reports_dir = f'{script_name}_reports'

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(prob._name)

        n2_filename = 'n2.html'
        scaling_filename = 'driver_scaling_report.html'
        coloring_filename = 'jacobian_to_compute_coloring.png'
        #
        p = pathlib.Path(problem_reports_dir).joinpath(n2_filename)
        self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(scaling_filename)
        self.assertTrue(p.is_file(),f'The scaling report file, {str(p)}, was not found')
        p = pathlib.Path(problem_reports_dir).joinpath(coloring_filename)
        self.assertTrue(p.is_file(),f'The coloring report file, {str(p)}, was not found')

    def test_report_generation_no_reports(self):
        prob = om.Problem(reports=False)
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
        script_name = pathlib.Path(sys.argv[-1]).stem
        reports_dir = f'{script_name}_reports'
        n2_filename = f'{prob._name}_N2.html'
        scaling_filename = f'{prob._name}_driver_scaling.html'

        p = pathlib.Path(reports_dir).joinpath(n2_filename)
        self.assertFalse(p.is_file(),f'The N2 report file, {str(p)} was found but should not have')
        p = pathlib.Path(reports_dir).joinpath(scaling_filename)
        self.assertFalse(p.is_file(),f'The scaling report file, {str(p)}, was found but should not have')

    def test_report_generation_set_reports_dir(self):
        prob = om.Problem(reports_dir="custom_reports_dir")
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
        script_name = pathlib.Path(sys.argv[-1]).stem
        reports_dir = 'custom_reports_dir'
        n2_filename = f'{prob._name}_N2.html'
        scaling_filename = f'{prob._name}_driver_scaling.html'

        p = pathlib.Path(reports_dir).joinpath(n2_filename)
        self.assertTrue(p.is_file(),f'The N2 report file, {str(p)} was not found')
        p = pathlib.Path(reports_dir).joinpath(scaling_filename)
        self.assertTrue(p.is_file(),f'The scaling report file, {str(p)}, was not found')

if __name__ == '__main__':
    unittest.main()
