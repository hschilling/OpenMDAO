# Plot logtime in realtime using bokeh and tail -f
# Tested with python 3.5 and bokeh 0.12.4
# OSX/Linux only
# usage:
# 1. run 'bokeh serve'
# 2. run 'python3.5 main.py logfile.csv'
# assumes a logfile.csv with format:
# min_ask,1489758134.150000,1077.00,1076.78,0.45
# max_bid,1489758139.660000,1076.56,1076.76,0.41
# min_ask,1489758142.076000,1076.95,1076.76,0.40

import sys
import datetime
import asyncio
import asyncio.subprocess


from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.client import push_session
from bokeh.plotting import figure, curdoc

def push(timestamp, fair_price):
    source.stream(dict(x=[timestamp], y=[fair_price]))


@asyncio.coroutine
def update():
    create = asyncio.create_subprocess_exec('tail', '-f', '-n' , '+1', sys.argv[-1],
                                            stdout=asyncio.subprocess.PIPE)
    proc = yield from create
    while True:
        # Read one line of output
        data = yield from proc.stdout.readline()
        line = data.decode('ascii').rstrip()
        line = line.split(', ')
        # line format:
        # label,unix-timestamp,fair_price,spread
        if line[0] == 'max_bid' or line[0] == 'min_ask':
            timestamp = datetime.datetime.fromtimestamp(float(line[1]))
            fair_price = float(line[2])
            push(timestamp, fair_price)

source = ColumnDataSource(data=dict(x=[], y=[]))
p = figure()
p.xaxis.formatter=DatetimeTickFormatter()
l = p.line(x='x', y='y', source=source)

# open a session to keep our local document in sync with server
session = push_session(curdoc(), session_id='main')

session.show(p) # open the document in a browser

loop = asyncio.get_event_loop()
loop.run_until_complete(update())