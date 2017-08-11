"""
Class definition for BaseRecorder, the base class for all recorders.
"""
from fnmatch import fnmatchcase
from six import StringIO

from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.general_utils import warn_deprecation
from openmdao.core.system import System
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver, NonlinearSolver
from openmdao.recorders.recording_iteration_stack import recording_iteration_stack, \
    get_formatted_iteration_coordinate
from openmdao.utils.mpi import MPI


class BaseRecorder(object):
    """
    Base class for all case recorders and is not a functioning case recorder on its own.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_outputs'] :  bool(True)
        Tells recorder whether to record the outputs of a System.
    options['record_inputs'] :  bool(False)
        Tells recorder whether to record the inputs of a System.
    options['record_residuals'] :  bool(False)
        Tells recorder whether to record the residuals of a System.
    options['record_derivatives'] :  bool(False)
        Tells recorder whether to record the derivatives of a System.
    options['record_desvars'] :  bool(True)
        Tells recorder whether to record the desvars of a Driver.
    options['record_responses'] :  bool(False)
        Tells recorder whether to record the responses of a Driver.
    options['record_objectives'] :  bool(False)
        Tells recorder whether to record the objectives of a Driver.
    options['record_constraints'] :  bool(False)
        Tells recorder whether to record the constraints of a Driver.
    options['record_abs_error'] :  bool(True)
        Tells recorder whether to record the absolute error of a Solver.
    options['record_rel_error'] :  bool(True)
        Tells recorder whether to record the relative error of a Solver.
    options['record_solver_output'] :  bool(False)
        Tells recorder whether to record the output of a Solver.
    options['record_solver_derivatives'] :  bool(False)
        Tells recorder whether to record the derivatives of a Solver.
    options['includes'] :  list of strings("*")
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings('')
        Patterns for variables to exclude in recording (processed after includes).

    Attributes
    ----------
    out : StringIO
        Output to the recorder.
    _counter : int
        A global counter for execution order, used in iteration coordinate.
    _filtered_driver : dict
        Filtered subset of driver variables to record, based on includes/excludes.
    _filtered_solver : dict
        Filtered subset of solver variables to record, based on includes/excludes.
    _filtered_system : dict
        Filtered subset of system variables to record, based on includes/excludes.
    _desvars_values : dict
        Driver desvar values, post-filtering, to be used by a derived recorder.
    _responses_values : dict
        Driver response values, post-filtering, to be used by a derived recorder.
    _objectives_values : dict
        Driver objectives values, post-filtering, to be used by a derived recorder.
    _constraints_values : dict
        Driver constraints values, post-filtering, to be used by a derived recorder.
    _inputs : dict
        System inputs values, post-filtering, to be used by a derived recorder.
    _outputs : dict
        System or Solver output values, post-filtering, to be used by a derived recorder.
    _resids : dict
        System or Solver residual values, post-filtering, to be used by a derived recorder.
    _abs_error : float
        Solver abs_error value, to be used by a derived recorder.
    _rel_error : float
        Solver abs_error value, to be used by a derived recorder.
    _iteration_coordinate : str
        The unique iteration coordinate of where an iteration originates.
    _parallel : bool
        Designates if the current recorder is parallel-recording-capable.
    """

    def __init__(self):
        """
        initialize.
        """
        self.options = OptionsDictionary()
        # Options common to all objects
        self.options.declare('record_metadata', type_=bool, desc='Record metadata', default=True)
        self.options.declare('includes', type_=list, default=['*'],
                             desc='Patterns for variables to include in recording')
        self.options.declare('excludes', type_=list, default=[],
                             desc='Patterns for vars to exclude in recording '
                                  '(processed post-includes)')

        # Old options that will be deprecated
        self.options.declare('record_unknowns', type_=bool, default=False,
                             desc='Deprecated option to record unknowns.')
        self.options.declare('record_params', type_=bool, default=False,
                             desc='Deprecated option to record params.',)
        self.options.declare('record_resids', type_=bool, default=False,
                             desc='Deprecated option to record residuals.')
        self.options.declare('record_derivs', type_=bool, default=False,
                             desc='Deprecated option to record derivatives.')
        # System options
        self.options.declare('record_outputs', type_=bool, default=True,
                             desc='Set to True to record outputs at the system level')
        self.options.declare('record_inputs', type_=bool, default=True,
                             desc='Set to True to record inputs at the system level')
        self.options.declare('record_residuals', type_=bool, default=True,
                             desc='Set to True to record residuals at the system level')
        self.options.declare('record_derivatives', type_=bool, default=False,
                             desc='Set to True to record derivatives at the system level')
        # Driver options
        self.options.declare('record_desvars', type_=bool, default=True,
                             desc='Set to True to record design variables at the driver level')
        self.options.declare('record_responses', type_=bool, default=False,
                             desc='Set to True to record responses at the driver level')
        self.options.declare('record_objectives', type_=bool, default=False,
                             desc='Set to True to record objectives at the driver level')
        self.options.declare('record_constraints', type_=bool, default=False,
                             desc='Set to True to record constraints at the driver level')
        # Solver options
        self.options.declare('record_abs_error', type_=bool, default=True,
                             desc='Set to True to record absolute error at the solver level')
        self.options.declare('record_rel_error', type_=bool, default=True,
                             desc='Set to True to record relative error at the solver level')
        self.options.declare('record_solver_output', type_=bool, default=False,
                             desc='Set to True to record output at the solver level')
        self.options.declare('record_solver_residuals', type_=bool, default=False,
                             desc='Set to True to record residuals at the solver level')

        self.out = None

        # global counter that is used in iteration coordinate
        self._counter = 0

        # dicts in which to keep the included items for recording
        self._filtered_driver = {}
        self._filtered_system = {}
        self._filtered_solver = {}

        # For passing values from the base recorder to actual recorders
        # For Drivers
        self._desvars_values = None
        self._responses_values = None
        self._objectives_values = None
        self._constraints_values = None

        # For Systems
        self._inputs = None
        self._outputs = None
        self._resids = None

        # For Solvers
        self._abs_error = 0.0
        self._rel_error = 0.0

        # For Drivers, Systems, and Solvers
        self._iteration_coordinate = None

        # By default, this is False, but it should be set to True
        # if the recorder will record data on each process to avoid
        # unnecessary gathering.
        self._parallel = False

    def startup(self, object_requesting_recording):
        """
        Prepare for a new run and calculate inclusion lists.

        Parameters
        ----------
        object_requesting_recording :
            Object to which this recorder is attached.
        """
        self._counter = 0

        # Deprecated options here, but need to preserve backward compatibility if possible.
        if self.options['record_params']:
            warn_deprecation("record_params is deprecated, please use record_inputs.")
            # set option to what the user intended.
            self.options['record_inputs'] = True

        if self.options['record_unknowns']:
            warn_deprecation("record_ is deprecated, please use record_outputs.")
            # set option to what the user intended.
            self.options['record_outputs'] = True

        if self.options['record_resids']:
            warn_deprecation("record_params is deprecated, please use record_residuals.")
            # set option to what the user intended.
            self.options['record_residuals'] = True

        # Compute the inclusion/exclusion lists

        if (isinstance(object_requesting_recording, System)):
            myinputs = myoutputs = myresiduals = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_inputs']:
                if object_requesting_recording._inputs:
                    myinputs = {n for n in object_requesting_recording._inputs._names
                                if self._check_path(n, incl, excl)}
            if self.options['record_outputs']:
                if object_requesting_recording._outputs:
                    myoutputs = {n for n in object_requesting_recording._outputs._names
                                 if self._check_path(n, incl, excl)}
                if self.options['record_residuals']:
                    myresiduals = myoutputs  # outputs and residuals have same names
            elif self.options['record_residuals']:
                if object_requesting_recording._residuals:
                    myresiduals = {n for n in object_requesting_recording._residuals._names
                                   if self._check_path(n, incl, excl)}

            self._filtered_system = {
                'i': myinputs,
                'o': myoutputs,
                'r': myresiduals
            }

        if (isinstance(object_requesting_recording, Driver)):
            mydesvars = myobjectives = myconstraints = myresponses = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_desvars']:
                mydesvars = {n for n in object_requesting_recording._designvars
                             if self._check_path(n, incl, excl)}

            if self.options['record_objectives']:
                myobjectives = {n for n in object_requesting_recording._objs
                                if self._check_path(n, incl, excl)}

            if self.options['record_constraints']:
                myconstraints = {n for n in object_requesting_recording._cons
                                 if self._check_path(n, incl, excl)}

            if self.options['record_responses']:
                myresponses = {n for n in object_requesting_recording._responses
                               if self._check_path(n, incl, excl)}

            self._filtered_driver = {
                'des': mydesvars,
                'obj': myobjectives,
                'con': myconstraints,
                'res': myresponses
            }

        if (isinstance(object_requesting_recording, Solver)):
            myoutputs = myresiduals = set()
            incl = self.options['includes']
            excl = self.options['excludes']

            if self.options['record_solver_residuals']:
                if isinstance(object_requesting_recording, NonlinearSolver):
                    residuals = object_requesting_recording._system._residuals
                else:  # it's a LinearSolver
                    residuals = object_requesting_recording._system._vectors['residual']['linear']
                myresiduals = {n for n in residuals
                               if self._check_path(n, incl, excl)}

            if self.options['record_solver_output']:
                if isinstance(object_requesting_recording, NonlinearSolver):
                    outputs = object_requesting_recording._system._outputs
                else:  # it's a LinearSolver
                    outputs = object_requesting_recording._system._vectors['output']['linear']
                myoutputs = {n for n in outputs
                             if self._check_path(n, incl, excl)}

            self._filtered_solver = {
                'out': myoutputs,
                'res': myresiduals
            }

    def _check_path(self, path, includes, excludes):
        """
        Calculate whether `path` should be recorded.

        Parameters
        ----------
        path : str
            path proposed to be recorded
        includes : list
            list of things to be included in recording list.
        excludes : list
            list of things to be excluded from recording list.

        Returns
        -------
        boolean
            True if path should be recorded, False if it's been excluded.
        """
        # First see if it's included
        for pattern in includes:
            if fnmatchcase(path, pattern):
                # We found a match. Check to see if it is excluded.
                for ex_pattern in excludes:
                    if fnmatchcase(path, ex_pattern):
                        return False
                return True

        # Did not match anything in includes.
        return False

    def record_metadata(self, object_requesting_recording):
        """
        Route the record_metadata call to the proper method.

        Parameters
        ----------
        object_requesting_recording: <object>
            The object that would like to record its metadata.
        """
        if self.options['record_metadata']:
            if isinstance(object_requesting_recording, Driver):
                self.record_metadata_driver(object_requesting_recording)
            elif isinstance(object_requesting_recording, System):
                self.record_metadata_system(object_requesting_recording)
            elif isinstance(object_requesting_recording, Solver):
                self.record_metadata_solver(object_requesting_recording)

    def record_metadata_driver(self, object_requesting_recording):
        """
        Record driver metadata.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver that would like to record its metadata.
        """
        raise NotImplementedError()

    def record_metadata_system(self, object_requesting_recording):
        """
        Record system metadata.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System that would like to record its metadata.
        """
        raise NotImplementedError()

    def record_metadata_solver(self, object_requesting_recording):
        """
        Record solver metadata.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver that would like to record its metadata.
        """
        raise NotImplementedError()

    def record_iteration(self, object_requesting_recording, metadata, **kwargs):
        """
        Route the record_iteration call to the proper method.

        Parameters
        ----------
        object_requesting_recording : object
            System, Solver, Driver in need of recording.
        metadata : dict, optional
            Dictionary containing execution metadata.
        **kwargs : keyword args
            Some implementations of record_iteration need additional args.
        """
        if not self._parallel:
            if MPI and MPI.COMM_WORLD.rank > 0:
                raise RuntimeError("Non-parallel recorders should not be recording on ranks > 0")

        self._counter += 1

        self._iteration_coordinate = get_formatted_iteration_coordinate()

        if isinstance(object_requesting_recording, System):
            self.record_iteration_system(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, Solver):
            self.record_iteration_solver(object_requesting_recording, metadata, **kwargs)
        else:
            raise ValueError("Recorders must be attached to Drivers, Systems, or Solvers.")

    def record_iteration_driver(self, object_requesting_recording, desvars, responses,
                                objectives, constraints, metadata):
        """
        Record an iteration using the Driver options.

        Parameters
        ----------
        object_requesting_recording : object
            The Driver in need of recording.
        metadata : dict, optional
            Dictionary containing execution metadata.
        desvars: dict
            The design variables of the Driver being recorded.
        responses: dict
            The responses of the Driver being recorded.
        objectives: dict
            The objectives of the Driver being recorded.
        constraints: dict
            The constraints of the Driver being recorded.
        """
        # TODO: this code and the same code in record_iteration should be in a separate method
        if not self._parallel:
            if MPI and MPI.COMM_WORLD.rank > 0:
                raise RuntimeError("Non-parallel recorders should not be recording on ranks > 0")

        self._counter += 1
        self._iteration_coordinate = get_formatted_iteration_coordinate()

        if self.options['record_desvars']:
            if self._filtered_driver:
                self._desvars_values = \
                    {name: desvars[name] for name in self._filtered_driver['des']}
            else:
                self._desvars_values = desvars
        else:
            self._desvars_values = None

        # Cannot handle responses yet
        # if self.options['record_responses']:
        #     if self._filtered_driver:
        #         self._responses_values = \
        #             {name: responses[name] for name in self._filtered_driver['res']}
        #     else:
        #         self._responses_values = responses
        # else:
        #     self._responses_values = None

        if self.options['record_objectives']:
            if self._filtered_driver:
                self._objectives_values = \
                    {name: objectives[name] for name in self._filtered_driver['obj']}
            else:
                self._objectives_values = objectives
        else:
            self._objectives_values = None

        if self.options['record_constraints']:
            if self._filtered_driver:
                self._constraints_values = \
                    {name: constraints[name] for name in self._filtered_driver['con']}
            else:
                self._constraints_values = constraints
        else:
            self._constraints_values = None

    def record_iteration_system(self, object_requesting_recording, metadata):
        """
        Record an iteration using system options.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        method : str
            The method that called record_iteration. One of '_apply_linear', '_solve_linear',
            '_apply_nonlinear,' '_solve_nonlinear'. Behavior varies based on from which function
            record_iteration was called.
        """
        stack_top = recording_iteration_stack[-1][0]
        method = stack_top.split('.')[-1]

        if method not in ['_apply_linear', '_apply_nonlinear', '_solve_linear',
                          '_solve_nonlinear']:
            raise ValueError("method must be one of: '_apply_linear, "
                             "_apply_nonlinear, _solve_linear, _solve_nonlinear'")

        if 'nonlinear' in method:
            inputs, outputs, residuals = object_requesting_recording.get_nonlinear_vectors()
        else:
            inputs, outputs, residuals = object_requesting_recording.get_linear_vectors()

        if self.options['record_inputs'] and inputs._names:
            self._inputs = {}
            if 'i' in self._filtered_system:
                # use filtered inputs
                for inp in self._filtered_system['i']:
                    if inp in inputs._names:
                        self._inputs[inp] = inputs._names[inp]
            else:
                # use all the inputs
                self._inputs = inputs._names
        else:
            self._inputs = None

        if self.options['record_outputs'] and outputs._names:
            self._outputs = {}

            if 'o' in self._filtered_system:
                # use outputs from filtered list.
                for out in self._filtered_system['o']:
                    if out in outputs._names:
                        self._outputs[out] = outputs._names[out]
            else:
                # use all the outputs
                self._outputs = outputs._names
        else:
            self._outputs = None

        if self.options['record_residuals'] and residuals._names:
            self._resids = {}

            if 'r' in self._filtered_system:
                # use filtered residuals
                for res in self._filtered_system['r']:
                    if res in residuals._names:
                        self._resids[res] = residuals._names[res]
            else:
                # use all the residuals
                self._resids = residuals._names
        else:
            self._resids = None

    def record_iteration_solver(self, object_requesting_recording, metadata, **kwargs):
        """
        Record an iteration using solver options.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        absolute : float
            The absolute error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        relative : float
            The relative error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        """
        # Go through the recording options of Solver to construct the entry to be inserted.
        if self.options['record_abs_error']:
            self._abs_error = kwargs.get('abs')
        else:
            self._abs_error = None

        if self.options['record_rel_error']:
            self._rel_error = kwargs.get('rel')
        else:
            self._rel_error = None

        if self.options['record_solver_output']:

            if isinstance(object_requesting_recording, NonlinearSolver):
                outputs = object_requesting_recording._system._outputs
            else:  # it's a LinearSolver
                outputs = object_requesting_recording._system._vectors['output']['linear']

            self._outputs = {}
            if 'out' in self._filtered_solver:
                for outp in outputs._names:
                    self._outputs[outp] = outputs._names[outp]
            else:
                self._outputs = outputs
        else:
            self._outputs = None

        if self.options['record_solver_residuals']:

            if isinstance(object_requesting_recording, NonlinearSolver):
                residuals = object_requesting_recording._system._residuals
            else:  # it's a LinearSolver
                residuals = object_requesting_recording._system._vectors['residual']['linear']

            self._resids = {}
            if 'res' in self._filtered_solver:
                for rez in residuals._names:
                    self._resids[rez] = residuals._names[rez]
            else:
                self._resids = residuals
        else:
            self._resids = None

    def close(self):
        """
        Cleanup the recorder.
        """
        pass
