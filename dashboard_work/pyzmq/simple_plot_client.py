import time
import random
import zmq

context = zmq.Context.instance()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://127.0.0.1:1235")

t = 0
y = 0

while True:
    time.sleep(1.0)
    t += 1
    y += random.normalvariate(0, 1)
    print("client send")
    pub_socket.send_pyobj(dict(x=[t], y=[y]))
