import asyncio
import random
from threading import Thread

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.driving import linear
from bokeh.layouts import column
# from aiohttp import ClientSession

# Create a new session for aiohttp
# session = ClientSession()


# Make Tornado use asyncio's event loop
# from tornado.platform.asyncio import AsyncIOMainLoop
#
# AsyncIOMainLoop().install()

# ColumnDataSource setup
source = ColumnDataSource(dict(time=[], value=[]))

# Set up the figure
# p = figure(plot_height=400, plot_width=800, title="Real-time Data Plot",
#            tools="xpan,xwheel_zoom,xbox_zoom,reset")
p = figure(title="Real-time Data Plot",
           tools="xpan,xwheel_zoom,xbox_zoom,reset")

# Add a line
p.line(x='time', y='value', alpha=0.8, line_width=2, color='blue', source=source)

# Function to get new data point
async def get_new_data():
    # async with session.get('http://your_api_endpoint') as resp:  # replace with your real API endpoint
    #     return await resp.text()

    await asyncio.sleep(1)
    return 1.234

@linear()
def update(step):
    # Asynchronously get new data and append to the source
    # new_data = asyncio.run(get_new_data())

    loop = asyncio.get_event_loop()
    new_data = loop.run_until_complete(get_new_data())

    source.stream(dict(time=[step], value=[float(new_data)]), rollover=200)


def blocking_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    step = 0
    while True:
        new_data = loop.run_until_complete(get_new_data())
        source.stream(dict(time=[step], value=[float(new_data)]), rollover=200)
        step += 1
        import time
        time.sleep(1)  # Sleep for 1 second

# Start the blocking task in a separate thread
Thread(target=blocking_task).start()









curdoc().add_root(column(p))
# curdoc().add_periodic_callback(update, 1000)  # 1000ms = 1s
