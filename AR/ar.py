import math

import numpy as np
import cv2
import sys
import time

import cv2.aruco

# import cameracalibration

# from cv2 import Rodrigues, cornerSubPix, flip

# import cameracalibration

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

font = cv2.FONT_HERSHEY_SIMPLEX

prev_frame_time = 0

new_frame_time = 0

armode = True
usecalibration = False
num = 0


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    x += 180
    x = x * 180 / np.pi
    y = y * 180 / np.pi
    z = z * 180 / np.pi

    return x, y, z


def myRotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    x = math.atan2(R[2, 1], R[2, 2])
    z = math.atan2(R[1, 0], R[0, 0])
    if (math.cos(z) == 0):
        y = math.atan2(-R[2, 0], R[1, 0] / math.sin(z))
    else:
        y = math.atan2(-R[2, 0], R[0, 0] / math.cos(z))

    x = x * 180 / np.pi
    y = y * 180 / np.pi
    z = z * 180 / np.pi
    x += 180

    return x, y, z


def rotation_angles(matrix):
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan(-r12 / r11)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)

def drawBoxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255), 3)

    img = cv2.drawContours(img, [imgpts[4:]], -1,(0,0,255),3)
    return img

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    cv2.aruco.ARUCO_CW_TOP_LEFT_CORNER

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    # parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    eulerAngles = (0, 0, 0)
    tvec = [[[0.001, 10, 0.001]]]
    rot_mat = np.array([[1.0, 0, 0, 1.0],
                               [0, 1.0, 0, 1.0],
                               [0, 0, 1.0, 1.0],
                               [0, 0, 0, 1.0]])

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)

            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners[i], (11,11), (-1,-1), criteria)

            cv2.aruco.drawDetectedMarkers(frame, corners)
            axisBoxes = np.float32(
                [[0, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0], [0.03, 0, 0], [0, 0, -0.03], [0, 0.03, -0.03], [0.03, 0.03, -0.03], [0.03, 0, - 0.03]])
            imgpts, jac = cv2.projectPoints(axisBoxes, rvec, tvec, matrix_coefficients, distortion_coefficients)

            frame = drawBoxes(frame, corners2, imgpts)

            rot_mat, _ = cv2.Rodrigues(rvec)

            # eulerAngles = rotationMatrixToEulerAngles(rot_mat)
            eulerAngles = myRotationMatrixToEulerAngles(rot_mat)
            # eulerAngles = rotationMatrixToEulerAngles(rot_mat)

            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            cv2.putText(frame, "Marker detected!", (7, 80), font, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "PositionX: " + tvec[0][0][0].astype(str), (7, 120), font, 1.5, (0, 0, 255), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, "PositionY: " + tvec[0][0][1].astype(str), (7, 160), font, 1.5, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, "PositionZ: " + tvec[0][0][2].astype(str), (7, 200), font, 1.5, (255, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, "RotationX: " + str(eulerAngles[0]), (7, 240), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "RotationY: " + str(eulerAngles[1]), (7, 280), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "RotationZ: " + str(eulerAngles[2]), (7, 320), font, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
    return frame, eulerAngles, tvec, rot_mat


aruco_type = "DICT_7X7_1000"

# arucoDict = cv2.aruco.Dictionary.readDictionary(ARUCO_DICT[aruco_type])

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)

# arucoParams = cv2.aruco.DetectorParameters_create()

intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
intrinsic_cameraMemory = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
distortion = np.array((-0.0, 0.0, 0, 0))
distortionMemory = np.array((-0.0, 0.0, 0, 0))

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('vid.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    break;

    ret, img = cap.read()

    # flip(img, 1, img)

    if armode:  # armode

        output, _, __ = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(output, "FPS " + fps, (7, 40), font, 1.5, (100, 255, 0), 3, cv2.LINE_AA)
    else:
        output = img

    # Image save
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and armode == False:  # wait for 's' key to save and exit
        cv2.imwrite('images/saved' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    if key == ord('e'):  # wait for 'e' key to switch armode
        if armode:
            armode = False
        else:
            armode = True

    if key == ord('t'):  # wait for 't' key to switch calibration
        if usecalibration:
            usecalibration = False
            distortion = np.array((-0.0, 0.0, 0, 0))
            intrinsic_camera = np.array(((0.0, 0, 0.0), (0, 0.0, 0.0), (0, 0, 1)))
            print("No calibration")
        else:
            usecalibration = True
            distortion = distortionMemory
            intrinsic_camera = intrinsic_cameraMemory
            img = cv2.imread('images/saved0.png')
            h, w = img.shape[:2]
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_camera, distortion, (w, h), 1, (w, h))
            print("Use calibration")

    if usecalibration:
        output = cv2.undistort(img, intrinsic_camera, distortion, None, newCameraMatrix)

    # if key == ord('r'):  # wait for 'r' key to run calibration
    #    intrinsic_cameraMemory, distortionMemory = cameracalibration.runCalibration()
    #    print(intrinsic_camera)

    cv2.imshow('Estimated Pose', output)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
