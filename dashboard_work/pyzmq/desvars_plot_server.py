from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio

doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1241")
socket.setsockopt(zmq.SUBSCRIBE, b"")

def update(new_data):
    global source
    print("in update")
    source.stream(new_data, rollover=50)


async def loop():
    print('in def loop')
    while True:
        new_data = await socket.recv_pyobj()
        print("received new data")
        if 'desvars' not in new_data:
            doc.add_next_tick_callback(partial(update, new_data))

# source = ColumnDataSource(data=dict(x=[0], y=[0]))
example_data = dict()
example_data['t'] = [0]
example_data['x'] = [0]
example_data['z'] = [0]
example_data['con_cmp1.con1'] = [0]
example_data['con_cmp2.con2'] = [0]
example_data['obj_cmp.obj'] = [0]
source = ColumnDataSource(data=example_data)

plot = figure(title="OpenMDAO optimization run",
              height=1500, width=1500,
              # x_range=(0, 20.0), y_range=(0, 30.0),
              # x_range=(0, 20.0)
              )
plot.xaxis.axis_label = "Time (seconds)"
plot.yaxis.axis_label = "Problems Vars"
plot.line(x='t', y='x', source=source, color="violet",legend_label='x')
plot.line(x='t', y='z', source=source, color="green",legend_label='z')
plot.line(x='t', y='con_cmp1.con1', source=source, color="blue",legend_label='con1')
plot.line(x='t', y='con_cmp2.con2', source=source, color="black",legend_label='con2')
plot.line(x='t', y='obj_cmp.obj', source=source, color="red",legend_label='obj')

doc.add_root(plot)
print("spawn_callback")
IOLoop.current().spawn_callback(loop)
