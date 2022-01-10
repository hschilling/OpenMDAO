"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

import pathlib

reports_registry = []
from collections import namedtuple

Report = namedtuple('Report', 'probname reporting_object method pre_or_post func desc kwargs')

def register_report(func, desc, method, pre_or_post, reporting_object, probname=None, **kwargs ):
    global reports_registry
    report = Report(probname, reporting_object, method, pre_or_post, func, desc, kwargs)
    reports_registry.append(report)
    return

def setup_default_reports():
    from openmdao.visualization.n2_viewer.n2_viewer import n2
    register_report(n2, 'create n2', 'final_setup', 'post', 'problem', probname=None, show_browser=False)
    from openmdao.visualization.scaling_viewer.scaling_report import view_driver_scaling
    register_report(view_driver_scaling, 'view_driver_scaling', 'final_setup', 'post', 'driver', probname=None,
                    show_browser=False)
    from openmdao.utils.coloring import coloring_reporting
    register_report(coloring_reporting, 'coloring_reporting', 'final_setup', 'post', 'problem', probname=None)


# TODO Support env var of OPENMDAO_REPORTS with values of 0, false, off to disable all report generation
# import inspect
# import pathlib
# script_path = inspect.stack()[-1][1]
# script_name = pathlib.Path(script_path).stem
# reports_dir = f'{script_name}_reports'
reports_dir = '.'

def set_reports_dir(reports_dir_path):
    global reports_dir
    reports_dir = reports_dir_path


# setup_default_reports()

def list_reports():
    pass

import os

def run_reports(prob, method, pre_or_post): # TODO can we use inspect to get method?
    global reports_dir

    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    reports_dir =  os.environ.get('OPENMDAO_REPORTS_DIR', reports_dir)

    # need to get the report directory
    if not os.path.isdir(reports_dir):
        os.mkdir(reports_dir)

    # loop through reports registry looking for matches
    for report in reports_registry:
        if report.probname and report.probname != prob._name:
            continue
        if report.method != method:
            continue
        if report.pre_or_post == pre_or_post:
            # TODO could use context mgr from here https://newbedev.com/how-can-i-change-directory-with-python-pathlib
            # make the problem reports dir
            problem_reports_dirname = f'{prob._name}_reports'
            problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(problem_reports_dirname)
            if not os.path.isdir(problem_reports_dirpath):
                os.mkdir(problem_reports_dirpath)
            prev_cwd = pathlib.Path.cwd()
            os.chdir(problem_reports_dirpath)
            if report.reporting_object == 'problem':
                reporting_object = prob
            elif report.reporting_object == 'driver':
                reporting_object = prob.driver
            report.func(reporting_object, **report.kwargs)
            os.chdir(prev_cwd)
