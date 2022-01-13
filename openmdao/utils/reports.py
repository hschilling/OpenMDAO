"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

from collections import namedtuple
import pathlib
import sys
import os
import contextlib
from pathlib import Path


from openmdao.visualization.scaling_viewer.scaling_report import view_driver_scaling

Report = namedtuple('Report', 'probname method pre_or_post func desc kwargs')

_reports_registry = []
_reports_dir = '.'  # the default location for the reports


def register_report(func, desc, method, pre_or_post, probname=None, **kwargs):
    global _reports_registry
    report = Report(probname, method, pre_or_post, func, desc, kwargs)
    _reports_registry.append(report)
    return


def view_driver_scaling_for_report(prob, **kwargs):
    """
    Created for the reporting system, which expects the reporting functions to have Problem as
    their first argument.

    Parameters
    ----------
    prob : Problem
        The problem used for the scaling report.
    **kwargs : dict
        Keyword args.

    Returns
    -------
    dict
        Data to used to generate html file.
    """
    return view_driver_scaling(prob.driver, **kwargs)


def setup_default_reports():
    from openmdao.visualization.n2_viewer.n2_viewer import n2
    register_report(n2, 'create n2', 'final_setup', 'post', probname=None, show_browser=False)
    register_report(view_driver_scaling_for_report, 'view_driver_scaling', 'final_setup', 'post',
                    probname=None,
                    show_browser=False)
    from openmdao.utils.coloring import coloring_reporting
    register_report(coloring_reporting, 'coloring_reporting', 'final_setup', 'post', probname=None)


setup_default_reports()


def set_reports_dir(reports_dir_path):
    global _reports_dir
    _reports_dir = reports_dir_path


def list_reports(out_stream=None):
    """
    Write table of variable names, values, residuals, and metadata to out_stream.

    Parameters
    ----------
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    """
    global _reports_registry

    if not out_stream:
        out_stream = sys.stdout

    column_names = ['probname', 'method', 'pre_or_post', 'desc', 'func']
    column_widths = {}
    # Determine the column widths of the data fields by finding the max width for all rows
    for column_name in column_names:
        column_widths[column_name] = len(column_name)

    for report in _reports_registry:
        for column_name in column_names:
            val = str(getattr(report, column_name))
            column_widths[column_name] = max(column_widths[column_name], len(val))

    out_stream.write("Here are the reports registered to run:\n\n")

    column_header = ''
    column_dashes = ''
    column_spacing = 2
    for i, column_name in enumerate(column_names):
        column_header += '{:{width}}'.format(column_name, width=column_widths[column_name])
        column_dashes += column_widths[column_name] * '-'
        if i < len(column_names) - 1:
            column_header += column_spacing * ' '
            column_dashes += column_spacing * ' '

    out_stream.write('\n')
    out_stream.write(column_header + '\n')
    out_stream.write(column_dashes + '\n')

    for report in _reports_registry:
        report_info = ''
        for i, column_name in enumerate(column_names):
            val = getattr(report, column_name)
            if column_name == 'func':
                val = val.__name__
            else:
                val = str(val)
            val_formatted = f"{val:<{column_widths[column_name]}}"
            report_info += val_formatted
            if i < len(column_names) - 1:
                report_info += column_spacing * ' '

        out_stream.write(report_info + '\n')

    out_stream.write('\n')


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def run_reports(prob, method, pre_or_post):
    global _reports_dir
    global _reports_registry

    # No running of reports when running under testflo
    if 'TESTFLO_RUNNING' in os.environ:
        return

    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', _reports_dir)

    # need to get the report directory
    if not os.path.isdir(reports_dir):
        os.mkdir(reports_dir)

    # loop through reports registry looking for matches
    for report in _reports_registry:
        if report.probname and report.probname != prob._name:
            continue
        if report.method != method:
            continue
        if report.pre_or_post == pre_or_post:
            # make the problem reports dir
            problem_reports_dirname = f'{prob._name}_reports'
            problem_reports_dirpath = pathlib.Path(reports_dir).joinpath(problem_reports_dirname)
            if not os.path.isdir(problem_reports_dirpath):
                os.mkdir(problem_reports_dirpath)

            with working_directory(problem_reports_dirpath):
                try:
                    report.func(prob, **report.kwargs)
                # Need to handle the coloring and scaling reports which can fail in this way
                #   because total Jacobian can't be computed
                except RuntimeError as err:
                    if str(err) != "Can't compute total derivatives unless " \
                                   "both 'of' or 'wrt' variables have been specified.":
                        raise err


def clear_reports():
    global _reports_registry
    _reports_registry = []
