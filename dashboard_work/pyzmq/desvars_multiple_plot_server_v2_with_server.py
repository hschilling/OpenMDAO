from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc


from bokeh.io import curdoc, show


import time

from bokeh.layouts import row
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio
import tornado


# from bokeh.io import curdoc
# from tornado.ioloop import IOLoop
# async def loop():
#     print('in def loop')
#     curdoc().title = "WatchList"
#
# IOLoop.current().spawn_callback(loop)

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

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1241")
socket.setsockopt(zmq.SUBSCRIBE, b"")

io_loop = IOLoop.current()

def build_doc(doc):
    print("in build_doc")

    source = ColumnDataSource(dict(x=[], y=[], avg=[]))
    doc.title = "WatchList"

    fig = figure()

    fig.line(source=source, x='x', y='y', line_width=2, alpha=.85, color='red')
    fig.line(source=source, x='x', y='avg', line_width=2, alpha=.85, color='blue')
    doc.add_root(fig)

    #
    # doc.title = "OM App"
    # var_names_by_type = {
    #     'designvars': ['x', 'z'],
    #     'cons': ['con_cmp1.con1', 'con_cmp2.con2'],
    #     'objs': ['obj_cmp.obj'],
    # }
    #
    # example_dict = dict()
    # for var_type, var_names_in_type in var_names_by_type.items():
    #     for var_name in var_names_in_type:
    #         example_dict[var_name] = [0]
    # print(f"{example_dict=}")
    # example_dict['t'] = [0]
    #
    # source = ColumnDataSource(data=example_dict)
    #
    # plots = []
    # for var_type, var_names_in_type in var_names_by_type.items():
    #     plot = figure(title=f"OpenMDAO optimization {var_type}", height=500, width=500)
    #     plot.xaxis.axis_label = "Time (seconds)"
    #     plot.yaxis.axis_label = f"{var_type} Vars"
    #     for var_name in var_names_in_type:
    #         plot.line(x='t', y=var_name, source=source, color="violet", legend_label=var_name)
    #     plots.append(plot)
    #
    # doc.add_root(row(*plots))

    def update(new_data):
        print("in update")
        source.stream(new_data, rollover=50)


    async def update_Data():

        doc.title = "WatchList"

        print('in update_Data')
        while True:
            new_data = await socket.recv_pyobj()
            print(f"received new data: {new_data}")
            if 'desvars' in new_data:  # need to do better here Semaphores??
                print(f"doc id desvars= {id(doc)}")
                # doc.add_next_tick_callback(partial(make_plots, new_data))
            else:
                print(f"doc id = {id(doc)}")
                doc.add_next_tick_callback(partial(update, new_data))

        source.stream(new_data, 100)

    doc.add_periodic_callback(update_Data, 100)


bokeh_app = Application(FunctionHandler(build_doc))

server = Server({'/': bokeh_app}, io_loop=io_loop)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
    io_loop.start()
