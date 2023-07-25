"""
Class definition for CaseRecorder, the base class for all recorders.
"""
import json
import time

import numpy as np
import zmq

from openmdao.core.driver import Driver
from openmdao.recorders.case_recorder import CaseRecorder
class PlottingRecorder(CaseRecorder):
    """
    Recorder that sends data to be plotted to the external process.
    """

    def __init__(self):
        """
        Initialize.
        """
        super().__init__()

    def startup(self, recording_requester, comm=None):
        """
        Prepare for a new run.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        comm : MPI.Comm or <FakeComm> or None
            The MPI communicator for the recorder (should be the comm for the Problem).
        """
        super().startup(recording_requester, comm)

        driver = recording_requester

        context = zmq.Context.instance()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://127.0.0.1:1241")

        # TODO need to do this in a better way to see if the socket is setup
        # apparently need some time for the socket to be setup before using it
        time.sleep(1)

        varnames_list_of_dict = dict()
        varnames_list_of_dict['desvars'] = list(driver._designvars.keys())
        varnames_list_of_dict['cons'] = list(driver._cons.keys())
        varnames_list_of_dict['objs'] = list(driver._objs.keys())

        with open("plotting_vars.txt", "w") as fp:
            json.dump(varnames_list_of_dict, fp)  # encode dict into JSON

    def record_viewer_data(self, model_viewer_data):
        pass

    def record_metadata_system(self, system, run_number=None):
        pass
    #
    def record_metadata_solver(self, solver, run_number=None):
        pass
    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        driver = recording_requester
        prob = driver._problem()

        prob_vars = prob.list_problem_vars()
        dict_of_values_to_send = dict()
        for var_type, list_of_tuples_in_type in prob_vars.items():
            for var_name, var_data in list_of_tuples_in_type:
                if var_data['size'] > 1:
                    value_to_plot = [np.linalg.norm(var_data['val'])]
                else:
                    value_to_plot = var_data['val']
                dict_of_values_to_send[var_name] = value_to_plot

        dict_of_values_to_send['t'] = [driver.iter_count]

        time.sleep(0.2) # seems to need a delay to get all the values

        self.pub_socket.send_pyobj(dict_of_values_to_send)
