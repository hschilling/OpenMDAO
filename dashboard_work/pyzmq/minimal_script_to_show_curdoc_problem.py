from bokeh.io import curdoc
from tornado.ioloop import IOLoop
async def loop():
    print('in def loop')
    curdoc().title = "WatchList"

IOLoop.current().spawn_callback(loop)

# https://docs.bokeh.org/en/2.4.3/docs/user_guide/server.html

# https://discourse.bokeh.org/t/document-not-registering-callback-initiated-from-asynchronous-socket-connection-handler/4570/4
# !!!!
# https://discourse.bokeh.org/t/integrate-bokeh-server-with-an-asyncio-api/2573/5