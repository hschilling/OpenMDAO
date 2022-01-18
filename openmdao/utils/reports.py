"""
Utility functions related to the reporting system which generates reports by default for all runs.
"""

from collections import namedtuple
import pathlib
import sys
import os

from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.utils.coloring import compute_total_coloring
from openmdao.utils.file_utils import working_directory

_Report = namedtuple('Report', 'func desc method pre_or_post probname kwargs')

_reports_registry = []
_reports_dir = '.'  # the default location for the reports

_Reports_Run = namedtuple('_Reports_Run', 'probname method pre_or_post' )
_reports_run = []

def register_report(func, desc, method, pre_or_post, probname=None, **kwargs):
    """
    Register a report with the reporting system.

    Parameters
    ----------
    func : function
        A function to do the reporting. Expects the first argument to be a Problem instance.
    desc : str
        A description of the report.
    method : str
        In which method of the Problem should this be run.
    pre_or_post : str
        Valid values are 'pre' and 'post'. Indicates when to run the report in the method.
    probname : str or None
        Either the name of a Problem or None. If None, then this report will be run for all
        Problems.
    **kwargs : dict
        Optional args for the reporting function.
    """
    global _reports_registry
    report = _Report(func, desc, method, pre_or_post, probname, kwargs)
    _reports_registry.append(report)
    return


def run_scaling_report(prob, **kwargs):
    """
    Run the scaling report.

    Created for the reporting system, which expects the reporting functions to have Problem as
    their first argument.

    Parameters
    ----------
    prob : Problem
        The problem used for the scaling report.
    **kwargs : dict
        Optional args for the scaling report function.
    """
    prob.driver.scaling_report(**kwargs)


def run_coloring_report(prob, **kwargs):
    """
    Run the coloring report.

    Created for the reporting system, which expects the reporting functions to have Problem as
    their first argument.

    Parameters
    ----------
    prob : Problem
        The problem used for the coloring report.
    **kwargs : dict
        Optional args for the coloring report function.
    """
    coloring = compute_total_coloring(prob, **kwargs)
    coloring.display(show=False)


def setup_default_reports():
    """
    Set up the default reports for all OpenMDAO runs.
    """
    # register_report(n2, 'N2 diagram', 'final_setup', 'post', probname=None, show_browser=False)
    # register_report(run_scaling_report, 'Driver scaling report', 'final_setup', 'post',
    #                 probname=None, show_browser=False)
    register_report(run_coloring_report, 'Coloring report', 'final_setup', 'post', probname=None)
    pass

def set_reports_dir(reports_dir_path):
    """
    Set the path to where the reports should go. Normally, they go into the current directory.

    Parameters
    ----------
    reports_dir_path : str
        Path to where the report directories should go.
    """
    global _reports_dir
    _reports_dir = reports_dir_path


def list_reports(out_stream=None):
    """
    Write table of information about reports currently registered in the reporting system.

    Parameters
    ----------
    out_stream : file-like object
        Where to send report info.
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


def run_reports(prob, method, pre_or_post):
    """
    Run all the registered reports.

    It takes into account the specifics of when and if
    they should be run at this point. This function is called from various methods of
    Problem.

    Parameters
    ----------
    prob : Problem
        OpenMDAO Problem instance.
    method : str
        Name of the method in Problem that this is called from.
    pre_or_post : str
        Where in the Problem method that this was called from. Only valid values are 'pre' and
        'post'.
    """
    global _reports_dir
    global _reports_registry

    # Keep track of what was run so we don't do it again. Prevents issues with recursion
    report_run = _Reports_Run(prob._name, method, pre_or_post)
    if report_run in _reports_run:
        return
    _reports_run.append(report_run)

    # No running of reports when running under testflo
    if 'TESTFLO_RUNNING' in os.environ:
        return

    # The user can define OPENMDAO_REPORTS to turn off reporting
    if 'OPENMDAO_REPORTS' in os.environ and os.environ['OPENMDAO_REPORTS'] in ['0', 'false', 'off']:
        return

    # The user can define where to put the reports using an environment variables
    reports_dir = os.environ.get('OPENMDAO_REPORTS_DIR', _reports_dir)

    # need to make the report directory if needed
    if os.path.isfile(reports_dir):
        raise RuntimeError(f"{_reports_dir} cannot be a reports directory because it is a file.")
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

            # with working_directory(problem_reports_dirpath):
            #     try:
            #         report.func(prob, **report.kwargs)
            #     # Need to handle the coloring and scaling reports which can fail in this way
            #     #   because total Jacobian can't be computed
            #     except RuntimeError as err:
            #         if str(err) != "Can't compute total derivatives unless " \
            #                        "both 'of' or 'wrt' variables have been specified.":
            #             raise err
            #
            # try:
            #     with working_directory(problem_reports_dirpath):
            #         report.func(prob, **report.kwargs)
            # # Need to handle the coloring and scaling reports which can fail in this way
            # #   because total Jacobian can't be computed
            # except RuntimeError as err:
            #     if str(err) != "Can't compute total derivatives unless " \
            #                        "both 'of' or 'wrt' variables have been specified.":
            #         raise err

            current_cwd = pathlib.Path.cwd()
            os.chdir(problem_reports_dirpath)
            try:
                report.func(prob, **report.kwargs)
            # Need to handle the coloring and scaling reports which can fail in this way
            #   because total Jacobian can't be computed
            except RuntimeError as err:
                if str(err) != "Can't compute total derivatives unless " \
                                   "both 'of' or 'wrt' variables have been specified.":
                    raise err
            finally:
                os.chdir(current_cwd)


def clear_reports():
    """
    Clear all of the reports from the registry.
    """
    global _reports_registry
    _reports_registry = []

def clear_reports_run():
    global _reports_run
    _reports_run = []

# When running under testflo, we don't want to waste time generating the reports and also
#   cluttering up the file system with reports
if 'TESTFLO_RUNNING' not in os.environ:
    setup_default_reports()
