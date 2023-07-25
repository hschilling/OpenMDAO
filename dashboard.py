import asyncio
import json
from threading import Thread
import os
import time
from datetime import datetime

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.palettes import Category10

import zmq.asyncio
import tornado

PLOTTING_FILE_NAME = 'plotting_vars.txt'

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1241")
socket.setsockopt(zmq.SUBSCRIBE, b"")

### get event loop from asyncio and use it in bokeh
from tornado.platform.asyncio import AsyncIOMainLoop
AsyncIOMainLoop().install()
io_loop = tornado.ioloop.IOLoop.current()

source = None

real_doc = None

start_time = datetime.now()

def get_file_creation_time(file_path):
    # Get file creation time in seconds since the epoch
    creation_time_sec = os.path.getctime(file_path)
    # Convert it to datetime object
    creation_time = datetime.fromtimestamp(creation_time_sec)
    return creation_time

def build_doc(doc):

    global source
    global real_doc
    real_doc = doc

    doc.title = "OpenMDAO Dashboard"

    # Wait for plotting vars file to be generated. Only open it if the time on it
    # is after the start time of this script - it is a hack!

    while True:
        # Does file exist?
        if os.path.exists(PLOTTING_FILE_NAME):
            file_time = get_file_creation_time(PLOTTING_FILE_NAME)
            difference = file_time - start_time

            if difference.total_seconds() > 0:
                with open(PLOTTING_FILE_NAME, "r") as fp:
                    # Load the dictionary from the file
                    var_names_by_type = json.load(fp)
                break
            time.sleep(1)

    example_dict = dict()
    for var_type, var_names_in_type in var_names_by_type.items():
        for var_name in var_names_in_type:
            example_dict[var_name] = [0]
    example_dict['t'] = [0]

    source = ColumnDataSource(data=example_dict)

    plots = []
    for var_type, var_names_in_type in var_names_by_type.items():
        plot = figure(title=f"OpenMDAO optimization {var_type}", height=500, width=500)
        # plot.xaxis.axis_label = "Time (seconds)"
        plot.xaxis.axis_label = "Iterations"
        plot.yaxis.axis_label = f"{var_type} Vars"
        color = iter(Category10[10])
        for var_name in var_names_in_type:
            plot.line(x='t', y=var_name, source=source, color=next(color), legend_label=var_name)
        plots.append(plot)

    doc.add_root(row(*plots))
async def update_Data():
    # Just having the while true here is causing the plots to not show up
    while True:
        new_data = await socket.recv_pyobj()
        if 'desvars' not in new_data:
            real_doc.add_next_tick_callback(lambda: source.stream(new_data, 100))

def blocking_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        new_data = loop.run_until_complete(update_Data())
        # source.stream(new_data, 100)
        # time.sleep(1)  # Sleep for 1 second

# Start the blocking task in a separate thread
Thread(target=blocking_task).start()

bokeh_app = Application(FunctionHandler(build_doc))

### get event loop from asyncio and use it in bokeh
# server = Server({'/': bokeh_app}, io_loop=io_loop)
server = Server({'/': bokeh_app})
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
    io_loop.start()
