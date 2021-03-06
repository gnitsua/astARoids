import time
from random import randint
import logging
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



    logging.basicConfig(level=logging.DEBUG)
    ps = PositionServer("5556")
    tracker = ArucoCornerTracker('drone.npz')
    client = VideoClient('192.168.1.1', 5555)
    client.connect()
    client.video_ready.wait()
    i = 0
    try:
        while True:

            try:
                cv2.imshow('im', client.frame)
                tvec = tracker.getCameraPosition(client.frame)
                print(
                    "x: %.2f cm, y: %.2f cm, z: %.2f cm, Pitch is %.2f degrees, Yaw is %.2f degrees, Roll is %.2f degrees." % (
                        tvec[0], tvec[1], tvec[2], tvec[3], tvec[4], tvec[5]))
                ps.sendPosition(tvec[0],  tvec[1], tvec[2])

            except RuntimeError:
                print("not enough markers")

            if cv2.waitKey(10) == ord('s'):
                print(client.frame)
                # png.from_array(client.frame, 'L').save("frame{}.png".format(i))
                cv2.imwrite('frame{}.png'.format(i), client.frame)
                i = i + 1
            elif cv2.waitKey(10) == ord(' '):
                break
    finally:
        client.close()
