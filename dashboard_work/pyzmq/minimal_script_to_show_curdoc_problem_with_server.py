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

io_loop = IOLoop.current()

curdoc().title = "WatchList"

def build_doc(doc):
    source = ColumnDataSource(dict(x=[], y=[], avg=[]))
    # doc.title = "WatchList"

    fig = figure()

    fig.line(source=source, x='x', y='y', line_width=2, alpha=.85, color='red')
    fig.line(source=source, x='x', y='avg', line_width=2, alpha=.85, color='blue')
    doc.add_root(fig)

    @count() #  nice decorator to provide a count integer that gets incremented each call
    def update_Data(ct):
        sine = np.sin(ct)
        sine_sum = sum(source.data['y']) + sine
        new_data = dict(x=[ct], y=[sine], avg=[sine_sum / (ct + 1)])
        source.stream(new_data, 100)

    # curdoc().title = "WatchList"
    doc.title = "WatchList"

    doc.add_periodic_callback(update_Data, 100)


bokeh_app = Application(FunctionHandler(build_doc))

server = Server({'/': bokeh_app}, io_loop=io_loop)
server.start()

if __name__ == '__main__':
    print('Opening Bokeh application on http://localhost:5006/')

    io_loop.add_callback(server.show, "/")
    io_loop.start()

