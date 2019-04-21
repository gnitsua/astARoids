import time
from random import randint

import cv2
import zmq as zmq
from pyardrone.video import VideoClient
from TrackMarkers.trackMarkers import ArucoCornerTracker

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
    tracker = ArucoCornerTracker('webcam.npz')
    client = VideoClient('192.168.1.1', 5555)
    client.connect()
    client.video_ready.wait()
    i = 0
    try:
        while True:
            cv2.imshow('im', client.frame)
            tvec = tracker.getCameraPosition(client.frame)
            print(
                "x: %.2f cm, y: %.2f cm, z: %.2f cm, Pitch is %.2f degrees, Yaw is %.2f degrees, Roll is %.2f degrees." % (
                    tvec[0] / 10.0, tvec[1] / 10.0, tvec[2] / 10.0, tvec[3], tvec[4], tvec[5]))
            # ps.sendPosition(x, y, z)
    finally:
        client.close()
