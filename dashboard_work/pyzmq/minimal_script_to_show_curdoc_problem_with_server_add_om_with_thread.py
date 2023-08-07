# from bokeh.io import curdoc
# from tornado.ioloop import IOLoop
# async def loop():
#     print('in def loop')
#     curdoc().title = "WatchList"
#
# IOLoop.current().spawn_callback(loop)
import _asyncio
import asyncio
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


# asyncio_loop = asyncio.new_event_loop()
# asyncio.set_event_loop(asyncio_loop)
#
async def get_plot_info(doc):
    print("get_plot_info start")
    new_data = await socket.recv_pyobj()

    print(f"{new_data=}")

    doc.add_next_tick_callback(lambda: create_plots_2(doc, new_data))

    print("get_plot_info end")
    return
#

def build_doc(doc):
    def setup_plot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # loop.run_until_complete(get_plot_info(doc))
        asyncio.run(get_plot_info(doc))
        print("end of setup_plot")
    Thread(target=setup_plot).start()
    time.sleep(10)
    print("end of build doc")

def build_doc_2(doc):
    print("inside build_doc")

    asyncio.run(get_plot_info(doc))
    #
    # loop = asyncio.get_event_loop()
    # # loop.run_until_complete(asyncio.gather(get_plot_info(doc)))
    #
    # print("create_task")
    # task = loop.create_task(get_plot_info(doc))
    # print("create_task done")
    # loop.run_until_complete(task)
    # print("run_until_complete done")

    # try:
    #
    #     # asyncio.run(get_plot_info(doc))
    #
    # except:
    #     pass # TODO need to handle RuntimeError: This event loop is already running

    # never reaches here!
    print("Done with build_doc")


def create_plots_2(doc, var_names_by_type):
    global source

    print("in create_plots_2")
    doc.title = "OM App"

    example_dict = dict()
    for var_type, var_names_in_type in var_names_by_type.items():
        for var_name in var_names_in_type:
            example_dict[var_name] = [0]
    example_dict['t'] = [0]


    print(f"{example_dict=}")

    print("creating source")
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

def build_doc_old(doc):
    print("in build_doc")

    global source
    global real_doc
    real_doc = doc


    print("in build_doc getting new_data")

    try:
        while True:
            try:
                # new_data = socket.recv(flags=zmq.NOBLOCK)
                # print("calling recv")
                new_data = socket.recv_pyobj()
                # print(f"{type(new_data)=}")
                if type(new_data) is not _asyncio.Future:
                    print("type(new_data) is not _asyncio.Future")
                    break
                # print(f"Received: {new_data}")
                # break  # Exit the inner loop if a message is received
            except zmq.error.Again:
                print("exception caught")
    except KeyboardInterrupt:
        print("socket subscriber stopped.")

    # if socket.poll(1000*10, zmq.POLLIN):
    #     new_data = socket.recv(zmq.NOBLOCK)
    #
    #     print("got message ", new_data)
    # else:
    #     print("error: message timeout")

    print(f"in build_doc got {new_data}")



    # print("get plot info")
    # new_data = asyncio_loop.run_until_complete(get_plot_info())

    create_plots(real_doc)

def create_plots(doc):
    global source

    print("in create_plots")
    doc.title = "OM App"
    var_names_by_type = {
        'designvars': ['x', 'z'],
        'cons': ['con_cmp1.con1', 'con_cmp2.con2'],
        'objs': ['obj_cmp.obj'],
    }

    example_dict = dict()
    for var_type, var_names_in_type in var_names_by_type.items():
        for var_name in var_names_in_type:
            example_dict[var_name] = [0]
    example_dict['t'] = [0]


    print(f"{example_dict=}")

    print("creating source")
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

def get_real_doc():
    global real_doc

    print(f"get_real_doc returning {real_doc}")
    return real_doc

async def update_Data():
    print("in update_Data")
    global source



    # Just having the while true here is causing the plots to not show up
    while True:
        new_data = await socket.recv_pyobj()
        print(f"received new data: {new_data}")
        actual_doc = get_real_doc()

        if 'desvars' not in new_data:
            print("source stream")
            actual_doc.add_next_tick_callback(lambda: source.stream(new_data, 100))
            # source.stream(new_data, 100)
        else:
            print("add_next_tick_callback create_plots")
            actual_doc.add_next_tick_callback(lambda: create_plots(actual_doc))



def blocking_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        new_data = loop.run_until_complete(update_Data())
        print('source stream in blocking task')
        source.stream(new_data, 100)
        import time
        time.sleep(1)  # Sleep for 1 second


bokeh_app = Application(FunctionHandler(build_doc))



### get event loop from asyncio and use it in bokeh
server = Server({'/': bokeh_app}, io_loop=io_loop)
# server = Server({'/': bokeh_app})
server.start()


if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")

    # Start the blocking task in a separate thread
    # Thread(target=blocking_task).start()



    io_loop.start()



