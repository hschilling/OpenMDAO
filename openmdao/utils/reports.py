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
    # from openmdao.utils.coloring import coloring_reporting
    # register_report(coloring_reporting, 'coloring_reporting', 'final_setup', 'post', 'problem', probname=None)


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
import sys

def list_reports(out_stream=None):
    """
    Write table of variable names, values, residuals, and metadata to out_stream.

    Parameters
    ----------
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    """
    global reports_registry

    if not out_stream:
        out_stream = sys.stdout

    column_names = ['probname', 'method', 'pre_or_post', 'desc', 'func']
    column_widths = {}
    # Determine the column widths of the data fields by finding the max width for all rows
    for column_name in column_names:
        column_widths[column_name] = len(column_name)

    for report in reports_registry:
        for column_name in column_names:
            val = str(getattr(report, column_name))
            column_widths[column_name] = max(column_widths[column_name], len(val))

    out_stream.write("Here are the reports registered to run:\n\n")


    # Write out the column headers
    # column_header = '{:{align}{width}}'.format('varname', align=align,
    #                                            width=max_varname_len)
    column_header = ''
    # column_dashes = max_varname_len * '-'
    column_dashes = ''
    column_spacing = 2
    for i, column_name in enumerate(column_names):
        column_header += '{:{width}}'.format(column_name, width=column_widths[column_name])
        # column_header += column_name
        column_dashes += column_widths[column_name] * '-'
        if i < len(column_names) - 1:
            column_header += column_spacing * ' '
            column_dashes += column_spacing * ' '

    out_stream.write('\n')
    out_stream.write(column_header + '\n')
    out_stream.write(column_dashes + '\n')

    for report in reports_registry:
        report_info = ''
        for i, column_name in enumerate(column_names):
            # report_info += '{:{width}}'.format(report[column_name], width=column_widths[column_name])
            val = getattr(report, column_name)
            if column_name == 'func':
                val = val.__name__
            else:
                val = str(val)
            val_formatted = f"{val:<{column_widths[column_name]}}"
            report_info += val_formatted
            # report_info += '{:{width}}'.format(val, width=column_widths[column_name])
            if i < len(column_names) - 1:
                report_info += column_spacing * ' '

        # print(f"{report.probname} {report.method} {report.pre_or_post} {report.desc} {report.func}")
        out_stream.write(report_info + '\n')

    out_stream.write('\n')

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
            else:
                raise ValueError(f"Invalid reporting object {report.reporting_object}.")
            report.func(reporting_object, **report.kwargs)
            os.chdir(prev_cwd)
