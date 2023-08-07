# from bokeh.io import curdoc
# from tornado.ioloop import IOLoop
# async def loop():
#     print('in def loop')
#     curdoc().title = "WatchList"
#
# IOLoop.current().spawn_callback(loop)
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

# https://docs.bokeh.org/en/2.4.3/docs/user_guide/server.html

# https://discourse.bokeh.org/t/document-not-registering-callback-initiated-from-asynchronous-socket-connection-handler/4570/4
# !!!!
# https://discourse.bokeh.org/t/integrate-bokeh-server-with-an-asyncio-api/2573/5

from bokeh.plotting import figure, curdoc

from bokeh.models import ColumnDataSource
import numpy as np
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.driving import count


from bokeh.plotting import figure, curdoc

from bokeh.models import ColumnDataSource
import numpy as np
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.driving import count

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc


from bokeh.io import curdoc, show


import time

from bokeh.layouts import row
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio
import tornado

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

def build_doc(doc):

    global source
    global real_doc
    real_doc = doc

    print("in build_doc")

    doc.title = "OM App"

    with open("plotting_vars.txt", "r") as fp:
        # Load the dictionary from the file
        var_names_by_type = json.load(fp)

    # var_names_by_type = {
    #     'designvars': ['x', 'z'],
    #     'cons': ['con_cmp1.con1', 'con_cmp2.con2'],
    #     'objs': ['obj_cmp.obj'],
    # }

    example_dict = dict()
    for var_type, var_names_in_type in var_names_by_type.items():
        for var_name in var_names_in_type:
            example_dict[var_name] = [0]
    print(f"{example_dict=}")
    example_dict['t'] = [0]


    print(f"{example_dict=}")

    source = ColumnDataSource(data=example_dict)

    plots = []
    for var_type, var_names_in_type in var_names_by_type.items():
        plot = figure(title=f"OpenMDAO optimization {var_type}", height=500, width=500)
        plot.xaxis.axis_label = "Time (seconds)"
        plot.yaxis.axis_label = f"{var_type} Vars"
        for var_name in var_names_in_type:
            plot.line(x='t', y=var_name, source=source, color="violet", legend_label=var_name)
        plots.append(plot)

    from bokeh.layouts import row

    print("plots to be added to root")
    doc.add_root(row(*plots))
    print("plots were added to root")

async def update_Data():
    print("in update_Data")
    x = 1.2
    z = 2.4
    con1 = 4.5
    con2 = 7.2
    obj = 3.1

    # new_data = {
    #     't' : [float(time.time())],
    #     'x' : [x],
    #     'z': [z],
    #     'con_cmp1.con1': [con1],
    #     'con_cmp2.con2': [con2],
    #     'obj_cmp.obj': [obj],
    # }

    # Just having the while true here is causing the plots to not show up
    while True:
        new_data = await socket.recv_pyobj()
        print(f"received new data: {new_data}")
        if 'desvars' not in new_data:
            print("source stream")
            real_doc.add_next_tick_callback(lambda: source.stream(new_data, 100))
            # source.stream(new_data, 100)


def blocking_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        new_data = loop.run_until_complete(update_Data())
        print('source stream in blocking task')
        source.stream(new_data, 100)
        import time
        time.sleep(1)  # Sleep for 1 second

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
