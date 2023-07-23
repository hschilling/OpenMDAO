from bokeh.document import without_document_lock
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio

print('-----------------------------starting-----------------------')

doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1240")
socket.setsockopt(zmq.SUBSCRIBE, b"")

## comment this out eventually
# source = ColumnDataSource(data=dict(t=[0], z=[0], x=[0], con1=[0], con2=[0], obj=[0]))


# new_data = socket.recv_pyobj()
#
# print(new_data)
#

var_names_by_type = {}


def update(new_data):
    source.stream(new_data, rollover=50)

@without_document_lock
async def loop():
    print('in def loop')
    while True:
        new_data = await socket.recv_pyobj()
        print(f"received new data: {new_data}")
        if 'desvars' in new_data:  # need to do better here Semaphores??
            var_names_by_type = new_data
            example_dict = dict()
            for var_type, var_names_in_type in var_names_by_type.items():
                for var_name in var_names_in_type:
                    example_dict[var_name] = [0]
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

            # this has good info
            # https://stackoverflow.com/questions/62488355/bokeh-multiple-live-streaming-graphs-in-different-objects-register-update-rout

            doc.add_root(row(*plots))
        else:
            doc.add_next_tick_callback(partial(update, new_data))


loop.nolock = True

async def loop_to_get_initial_data():
    print('in loop_to_get_initial_data')
    var_names_by_type = await socket.recv_pyobj()
    print("received initial data")

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

print("spawn_callback")
IOLoop.current().spawn_callback(loop)
