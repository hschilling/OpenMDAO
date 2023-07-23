from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc



from bokeh.io import curdoc, show

import asyncio

import time

from bokeh.layouts import row
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio
import tornado

print('-----------------------------starting-----------------------')

doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1241")
socket.setsockopt(zmq.SUBSCRIBE, b"")


var_names_by_type = {}

def update(new_data):
    global source
    print("in update")
    source.stream(new_data, rollover=50)

def make_plots(var_names_by_type):
    print("In make_plots")
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
    # doc = curdoc()
    print(f"doc id = {id(doc)}")

    # doc.add_root(row(*plots))
    print("after doc add root")


# @without_document_lock
async def loop():
    # doc = curdoc()

    print('in def loop')
    while True:
        new_data = await socket.recv_pyobj()
        print(f"received new data: {new_data}")
        if 'desvars' in new_data:  # need to do better here Semaphores??
            print(f"doc id desvars= {id(doc)}")
            doc.add_next_tick_callback(partial(make_plots, new_data))
        else:
            print(f"doc id = {id(doc)}")
            doc.add_next_tick_callback(partial(update, new_data))

async def modify_doc(doc):
    new_data = asyncio.gather(socket.recv_pyobj())
    print(f"received new data: {new_data}")
    if 'desvars' in new_data:  # need to do better here Semaphores??
        print(f"doc id desvars= {id(doc)}")
        doc.add_next_tick_callback(partial(make_plots, new_data))
    else:
        print(f"doc id = {id(doc)}")
        doc.add_next_tick_callback(partial(update, new_data))


from bokeh.server.server import Server
from tornado.ioloop import IOLoop

if __name__ == '__main__':
    server = Server({'/': modify_doc}, io_loop=IOLoop())
    server.start()
    server.io_loop.start()



# IOLoop.current().spawn_callback(loop)

# This might help?
#   https://www.tornadoweb.org/en/stable/ioloop.html

# Maybe make a class to capture everything ?
#     https://github.com/holoviz/holoviews/issues/3454#issuecomment-460849748

# Don't use bokeh serve, make server manually
#    https://stackoverflow.com/questions/55891252/how-to-make-bokeh-server-work-with-name-main

from bokeh.server.server import Server

# def start_loop():
#     print("in start loop")
#     server = Server(curdoc())
#     print("server.start")
#     server.start()
#     print("IOLoop.current().spawn_callback")
#     IOLoop.current().spawn_callback(loop)
#     show(server)
#
#
# if __name__ == '__main__':
#     print("in main")
#     start_loop()




# loop.nolock = True

# async def loop_to_get_initial_data():
#     print('in loop_to_get_initial_data')
#     var_names_by_type = await socket.recv_pyobj()
#     print("received initial data")

# IOLoop.current().spawn_callback(loop_to_get_initial_data)

# print("calling recv")
# message = socket.recv()
#
# # Process the received message
# print("Received message:", message.decode())
#
#
# import time
# while True:
#     try:
#         print("try getting var_names_by_type")
#         var_names_by_type = socket.recv_pyobj(0)
#         print(f'>>>>>>> {var_names_by_type=}')
#         break
#     except zmq.error.Again:
#         time.sleep(0.001)
#         print("zmq.error.Again")
#     except:
#         print("not captured exception")

    # except KeyboardInterrupt:
    #     break



#### restore this !!
# example_dict = dict()
# for var_type, var_names_in_type in var_names_by_type.items():
#     for var_name in var_names_in_type:
#         example_dict[var_name] = [0]
# print(f"{example_dict=}")
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

#####




# source = ColumnDataSource(data=dict(x=[0], y=[0]))
# source = ColumnDataSource(data=dict(t=[0], z=[0], x=[0], con1=[0], con2=[0], obj=[0]))
#
# desvars_plot = figure(title="OpenMDAO optimization desvars", height=500, width=500)
# desvars_plot.xaxis.axis_label = "Time (seconds)"
# desvars_plot.yaxis.axis_label = "Design Vars"
# desvars_plot.line(x='t', y='x', source=source, color="violet",legend_label='x')
# desvars_plot.line(x='t', y='z', source=source, color="green",legend_label='z')
# # doc.add_root(desvars_plot)
#
# cons_plot = figure(title="OpenMDAO optimization constraints", height=500, width=500)
# cons_plot.xaxis.axis_label = "Time (seconds)"
# cons_plot.yaxis.axis_label = "Constraints"
# cons_plot.line(x='t', y='con1', source=source, color="blue",legend_label='con1')
# cons_plot.line(x='t', y='con2', source=source, color="black",legend_label='con2')
# # doc.add_root(cons_plot)
#
# obj_plot = figure(title="OpenMDAO optimization objective", height=500, width=500)
# obj_plot.xaxis.axis_label = "Time (seconds)"
# obj_plot.yaxis.axis_label = "Objective"
# obj_plot.line(x='t', y='obj', source=source, color="red",legend_label='obj')
# # doc.add_root(obj_plot)

# doc.add_root(row(desvars_plot, cons_plot, obj_plot))

# plot = figure(title=f"test", height=500, width=500)
# plot.xaxis.axis_label = "Time (seconds)"
# plot.yaxis.axis_label = f"stuff"
# doc.add_root(plot)

