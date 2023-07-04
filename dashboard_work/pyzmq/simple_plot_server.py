from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from functools import partial
from tornado.ioloop import IOLoop
import zmq.asyncio

doc = curdoc()

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1234")
socket.setsockopt(zmq.SUBSCRIBE, b"")

def update(new_data):
    source.stream(new_data, rollover=50)


async def loop():
    print('in def loop')
    while True:
        new_data = await socket.recv_pyobj()
        print("received new data")
        doc.add_next_tick_callback(partial(update, new_data))

source = ColumnDataSource(data=dict(x=[0], y=[0]))

plot = figure(height=500, width=500, x_range=(0, 20.0), y_range=(0, 30.0))
plot.line(x='x', y='y', source=source)

doc.add_root(plot)
print("spawn_callback")
IOLoop.current().spawn_callback(loop)
