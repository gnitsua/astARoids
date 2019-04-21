# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import matplotlib.pyplot as plt
import glob

def find_intersection(top,bottom,right):
	#Use the sum of two vector to find the intersection point
	#Given a,b,c, the sum of vector is b-a+c-a, the intersection point is b-a+c-a+a=b+c-a
	intersection=(bottom[0]+right[0]-top[0],bottom[1]+right[1]-top[1])
	return intersection

def rotationMatrix2Angle(R):
	x=math.atan2(R[2][1],R[2][2])/2/math.pi*360.0
	y=math.atan2(-R[2][0],math.sqrt(R[2][1]*R[2][1]+R[2][2]*R[2][2]))/2/math.pi*360.0
	z=math.atan2(R[1][0],R[0][0])/2/math.pi*360.0
	return [x,y,z]

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_1000)

dist_coef = np.zeros(4)
qr3d = np.float32([[-50, 50, 0], [-50, -50, 0], [50, -50, 0], [50, 50, 0]])
camera_frame = [[0,0,0], [40,0,0], [40,40,0], [0,40,0], [0,0,0]]
img = cv2.imread('calib_images/opencv_frame_10.png')
h, w = img.shape[:2]
npzfile = numpy.load('calib.npz')
mtx = npzfile['mtx']
dist = npzfile['dist']
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

K = newCameraMtx
# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
    markersX=2,
    markersY=2,
    markerLength=0.09,
    markerSeparation=0.01,
    dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None

cam = cv2.VideoCapture(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []


while (cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()

    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        # Make sure all 5 markers were detected before printing them out
        if ids is not None and (len(ids) > 2 and len(ids) < 4):
            # Print corners and ids to the console
            toplefts = {}
            print(ids)
            for i, corner in zip(ids, corners):
                # print('ID: {}; Corners: {}'.format(i, corner))
                toplefts[i[0]] = corner[0][0]
            try:
                top = toplefts[0]
                bottom = toplefts[2]
                right = toplefts[1]
                print("top: " + str(top) +", bottom: " + str(bottom) + "right: " + str(right))
                N = find_intersection(top,bottom,right)
                qr_im = np.float32([top,bottom,[N[0],N[1]],right])
                # Solve the PnP problem using OpenCV function and get the rotation and translation
                ret, rvec, tvec = cv2.solvePnP(qr3d, qr_im, K, dist_coef)
                # Change rotation vector to rotation matrix
                rot_mat, _ = cv2.Rodrigues(rvec)
                # Change rotation and translation from camera coordinate to world coordinate
                rot_mat = rot_mat.transpose()
                tvec = -np.dot(rot_mat, tvec)
                # draw the camera frame in the figure
                x = []
                y = []
                z = []

                for point in camera_frame:
                    point = np.array(point)
                    out = np.dot(rot_mat, point.transpose()) + tvec.transpose()
                    out = out[0]
                    x.append(out[0])
                    y.append(out[1])
                    z.append(out[2])



                # Calculate the rotation angle from rotation matrix
                angle = rotationMatrix2Angle(rot_mat)
                print("x: %.2f cm, y: %.2f cm, z: %.2f cm, Pitch is %.2f degrees, Yaw is %.2f degrees, Roll is %.2f degrees." %(tvec[0]/10.0,tvec[1]/10.0,tvec[2]/10.0,angle[0],angle[1],angle[2]))
            except KeyError:
                pass
        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            # Wait on this frame
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()