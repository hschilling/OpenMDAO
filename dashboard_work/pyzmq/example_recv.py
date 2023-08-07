import zmq

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://127.0.0.1:5555")

import zmq.asyncio
context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:1239")
socket.setsockopt(zmq.SUBSCRIBE, b"")

global var_names_by_type
async def loop_to_get_initial_data():
    print('in loop_to_get_initial_data')
    var_names_by_type = await socket.recv_pyobj()
    print("received initial data")

from tornado.ioloop import IOLoop

IOLoop.current().spawn_callback(loop_to_get_initial_data)

#
# while True:
#     # Wait for a message to be received
#     print("calling recv")
#     message = socket.recv()
#
#     # Process the received message
#     print("Received message:", message.decode())
#
#     # Send a reply
#     reply = "Message received"
#     socket.send(reply.encode())
