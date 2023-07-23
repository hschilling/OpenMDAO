# from bokeh.io import curdoc
# from tornado.ioloop import IOLoop
# async def loop():
#     print('in def loop')
#     curdoc().title = "WatchList"
#
# IOLoop.current().spawn_callback(loop)
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

# io_loop = IOLoop.current()



### get event loop from asyncio and use it in bokeh
from tornado.platform.asyncio import AsyncIOMainLoop
AsyncIOMainLoop().install()
io_loop = tornado.ioloop.IOLoop.current()

# server = Server({'/': bokeh_app}, io_loop=io_loop)
# server.start()


# curdoc().title = "WatchList"

def build_doc(doc):

    print("in build_doc")

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

    from bokeh.layouts import row

    print("plots to be added to root")
    doc.add_root(row(*plots))
    print("plots were added to root")


    # source = ColumnDataSource(dict(x=[], y=[], avg=[]))
    # # doc.title = "WatchList"
    #
    # fig = figure()
    #
    # fig.line(source=source, x='x', y='y', line_width=2, alpha=.85, color='red')
    # fig.line(source=source, x='x', y='avg', line_width=2, alpha=.85, color='blue')
    # doc.add_root(fig)

    # @count() #  nice decorator to provide a count integer that gets incremented each call
    # async def update_Data(ct):
    # async def update_Data():

    from tornado import gen

    @gen.coroutine
    def non_blocking():
        # print("in non_blocking")
        yield IOLoop.current().run_in_executor(None, update_Data)

    def update():
        # print("in update")
        non_blocking()  # Runs the async task periodically

    def wrapper():
        import asyncio
        asyncio.run(update_Data())
        # update_Data()

    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    import concurrent.futures

    def run_async_task():
        new_loop = asyncio.new_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(start_loop, new_loop)
        asyncio.run_coroutine_threadsafe(update_Data(), new_loop)

    async def update_Data():

        print("in update_Data")

        # sine = np.sin(ct)
        # sine_sum = sum(source.data['y']) + sine
        # new_data = dict(x=[ct], y=[sine], avg=[sine_sum / (ct + 1)])

        x = 1.2
        z = 2.4
        con1 = 4.5
        con2 = 7.2
        obj = 3.1

        new_data = {
            't' : [float(time.time())],
            'x' : [x],
            'z': [z],
            'con_cmp1.con1': [con1],
            'con_cmp2.con2': [con2],
            'obj_cmp.obj': [obj],
        }

        pass

        # Just having the while true here is causing the plots to not show up
        while True:
            new_data = await socket.recv_pyobj()
            print(f"received new data: {new_data}")
        #
        source.stream(new_data, 100)

    # curdoc().title = "WatchList"

    # if you remove this the plots show up
    # doc.add_periodic_callback(update_Data, 100)
    doc.add_periodic_callback(update, 100)   ### at least the plots show up here
    # doc.add_periodic_callback(wrapper, 100) # get this error but plots show up
            # RuntimeError: asyncio.run() cannot be called from a running event loop

    # plots show up but update never gets run and was not awaited
    # doc.add_periodic_callback(lambda: IOLoop.current().run_in_executor(ThreadPoolExecutor(), wrapper), 100)

    # ??? https://stackoverflow.com/questions/48725890/runtimeerror-there-is-no-current-event-loop-in-thread-thread-1-multithreadi

    # This might have some great info!
    #     https://discourse.holoviz.org/t/can-i-load-data-asynchronously-in-panel/452

    # Some people say should use Dask
    #    https://docs.dask.org/en/stable/deploying-python-advanced.html

    # More async ideas
    #    https://danielmuellerkomorowska.com/2022/02/14/measuring-and-visualizing-gpu-power-usage-in-real-time-with-asyncio-and-matplotlib/

    # This is VERY interesting
    # https://groups.io/g/insync/message/4098

    # doc.add_periodic_callback(run_async_task, 100)
    # lambda: IOLoop.current().run_in_executor(ThreadPoolExecutor(), wrapper)


bokeh_app = Application(FunctionHandler(build_doc))

### get event loop from asyncio and use it in bokeh
# server = Server({'/': bokeh_app}, io_loop=io_loop)
server = Server({'/': bokeh_app})
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
    io_loop.start()

