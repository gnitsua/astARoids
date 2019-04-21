import time
from random import randint

import zmq as zmq


class PositionServer():
    def __init__(self, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % port)

    def sendPosition(self, x, y, z):
        topic = 10001
        self.socket.send_string("%d %s" % (topic, "(%d,%d,%d)" % (x, y, z)))


if __name__ == "__main__":
    ps = PositionServer("5556")

    while (True):
        x = randint(100, 1000)
        y = randint(100, 1000)
        z = randint(100, 1000)
        ps.sendPosition(x, y, z)
        print("(%d,%d,%d)" % (x, y, z))
        time.sleep(0.5)
