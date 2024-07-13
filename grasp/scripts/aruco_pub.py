#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
import cv2
import pyrealsense2 as rs
import numpy as np

def get_marker_info(pipeline, marker_length, camera_matrix, dist_coeffs, dictionary, parameters, base_marker_id, object_marker_id, last_known_Q):
    Q = np.zeros(3)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return last_known_Q

    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None and base_marker_id in ids and object_marker_id in ids:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        base_idx = np.where(ids == base_marker_id)[0][0]
        object_idx = np.where(ids == object_marker_id)[0][0]

        rvec_base = rvecs[base_idx][0]
        tvec_base = tvecs[base_idx][0]

        rvec_object = rvecs[object_idx][0]
        tvec_object = tvecs[object_idx][0]

        rotation_matrix_base, _ = cv2.Rodrigues(rvec_base)
        rotation_matrix_object, _ = cv2.Rodrigues(rvec_object)

        relative_rotation_matrix = np.dot(rotation_matrix_base.T, rotation_matrix_object)
        relative_translation_vector = np.dot(rotation_matrix_base.T, tvec_object - tvec_base)

        yaw = np.arctan2(relative_rotation_matrix[1, 0], relative_rotation_matrix[0, 0])

        Q[:] = [relative_translation_vector[0],
                relative_translation_vector[1],
                yaw]

        Q = np.round(Q, 5)
        last_known_Q = Q
    else:
        Q = last_known_Q

    cv2.imshow('Video Feed', frame)
    key = cv2.waitKey(1)
    if key == 27:  
        pipeline.stop()
        cv2.destroyAllWindows()

    return Q

def main():
    rospy.init_node('aruco_publisher', anonymous=True)
    pub = rospy.Publisher('aruco_data', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10) 

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    marker_length = 0.035
    camera_matrix = np.array([[593.41827166, 0, 313.63984994],
                              [0, 593.62545055, 251.75863783],
                              [0, 0, 1]])
    dist_coeffs = np.array([0.0130745949, 0.646725640, 0.00203177405, 0.000309401928, -1.95934330])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    base_marker_id = 0
    object_marker_id = 1
    last_known_Q = np.zeros(3)

    try:
        rospy.loginfo("Publishing Object Pose (via ArUco)...")
        while not rospy.is_shutdown():
            Q = get_marker_info(pipeline, marker_length, camera_matrix, dist_coeffs, dictionary, parameters, base_marker_id, object_marker_id, last_known_Q)
            msg = Float64MultiArray(data=Q.tolist())
            pub.publish(msg)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
