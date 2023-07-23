from bokeh.document import without_document_lock
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio
import tornado

print('-----------------------------starting-----------------------')

# doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1241")
socket.setsockopt(zmq.SUBSCRIBE, b"")



def make_plots(var_names_by_type):
    global source
    example_dict = dict()
    # TODO - do this more efficiently
    for var_type, var_names_in_type in var_names_by_type.items():
        for var_name in var_names_in_type:
            example_dict[var_name] = [0]
    print(f"{example_dict=}")
    example_dict['t'] = [0]
    source = ColumnDataSource(data=example_dict)

    plots = []
    for var_type, var_names_in_type in var_names_by_type.items():
        plot = figure(title=f"OpenMDAO optimization {var_type}", height=500, width=500)
        plot.xaxis.axis_label = "Time (seconds)"
        plot.yaxis.axis_label = f"{var_type} Vars"
        for var_name in var_names_in_type:
            plot.line(x='t', y=var_name, source=source, color="violet", legend_label=var_name)
        plots.append(plot)

    # this has good info
    # https://stackoverflow.com/questions/62488355/bokeh-multiple-live-streaming-graphs-in-different-objects-register-update-rout

    print(f"*****{plots=}")
    doc = curdoc()
    doc.add_root(row(*plots))

var_names_by_type = {
    'designvars': ['x'],
    'cons': ['cons1'],
    'objs': ['obj'],
}

make_plots(var_names_by_type)

import time
async def loop():
    print('in def loop')
    while True:
        await time.sleep(.2)

IOLoop.current().spawn_callback(loop)
