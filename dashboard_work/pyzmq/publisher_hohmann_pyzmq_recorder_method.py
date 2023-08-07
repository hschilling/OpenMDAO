#!/usr/bin/env python
# coding: utf-8

from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
import openmdao.api as om

from openmdao.recorders.plotting_recorder import PlottingRecorder

from openmdao.test_suite.test_examples.test_hohmann_transfer import VCircComp, TransferOrbitComp, DeltaVComp
import numpy as np

prob = om.Problem()

model = prob.model

model.add_subsystem('leo', subsys=VCircComp(), promotes_inputs=[('r', 'r1'), 'mu'])
model.add_subsystem('geo', subsys=VCircComp(), promotes_inputs=[('r', 'r2'), 'mu'])

model.add_subsystem('transfer', subsys=TransferOrbitComp(),
                    promotes_inputs=[('rp', 'r1'), ('ra', 'r2'), 'mu'])

model.add_subsystem('dv1', subsys=DeltaVComp(), promotes_inputs=[('dinc', 'dinc1')])

model.connect('leo.vcirc', 'dv1.v1')
model.connect('transfer.vp', 'dv1.v2')

model.add_subsystem('dv2', subsys=DeltaVComp(), promotes_inputs=[('dinc', 'dinc2')])

model.connect('transfer.va', 'dv2.v1')
model.connect('geo.vcirc', 'dv2.v2')

model.add_subsystem('dv_total',
                    subsys=om.ExecComp('delta_v=dv1+dv2',
                                       delta_v={'units': 'km/s'},
                                       dv1={'units': 'km/s'},
                                       dv2={'units': 'km/s'}),
                    promotes=['delta_v'])

model.connect('dv1.delta_v', 'dv_total.dv1')
model.connect('dv2.delta_v', 'dv_total.dv2')

model.add_subsystem('dinc_total',
                    subsys=om.ExecComp('dinc=dinc1+dinc2',
                                       dinc={'units': 'deg'},
                                       dinc1={'units': 'deg'},
                                       dinc2={'units': 'deg'}),
                    promotes=['dinc', 'dinc1', 'dinc2'])

prob.driver = om.ScipyOptimizeDriver()

model.add_design_var('dinc1', lower=0, upper=28.5)
model.add_design_var('dinc2', lower=0, upper=28.5)
model.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
model.add_objective('delta_v', scaler=1.0)

# set defaults for our promoted variables to remove ambiguities in value and/or units
model.set_input_defaults('r1', val=6778.0)
model.set_input_defaults('r2', val=42164.0)
model.set_input_defaults('mu', val=398600.4418)
model.set_input_defaults('dinc1', val=0., units='deg')
model.set_input_defaults('dinc2', val=28.5, units='deg')

# Setup the problem
prob.setup()

driver = prob.driver
recorder = PlottingRecorder()
driver.add_recorder(recorder)

prob.run_driver()
