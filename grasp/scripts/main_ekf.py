'''
Proprioception based EKF implementation

NOTE:
    The process model (state transition function) 'f' uses joint angles and velocities to predict the next state Y = [x1 y1 x2 y2]
    therefore, the jacobain matrix F = df/dY will be an indentity matrix. And, since the measurement model 'h' got the predicted measurements using joint angles which is not in the state Y, the 
    jacobian matrix H = dh/dY will also be an identity matrix. 
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import rospy
import time
import numpy as np
from numpy import sqrt, pi, cos, sin, arctan2, array 
from sympy import symbols, Matrix   
from std_msgs.msg import Float64MultiArray
from motorControl import MultimotorControl

'''
UTILITY FUNCTIONS
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def grasp(a, theta):
    return array([[-sin(theta), -cos(theta), sin(theta), cos(theta)], 
        [cos(theta), -sin(theta), -cos(theta), sin(theta)], 
        [a*sin(theta)**2 + a*cos(theta)**2, 0, a*sin(theta)**2 + a*cos(theta)**2, 0]])
    
def handJacobian(theta, L, phi):
    link1 = L[0]
    link2 = L[1]
    q1 = phi[0]
    q2 = phi[1]
    q3 = phi[2]
    q4 = phi[3]
    return array([[-(-link1*sin(q1) - link2*sin(q1 + q2))*sin(theta) + (link1*cos(q1) + link2*cos(q1 + q2))*cos(theta), link2*sin(theta)*sin(q1 + q2) + link2*cos(theta)*cos(q1 + q2), 0, 0], 
        [-(-link1*sin(q1) - link2*sin(q1 + q2))*cos(theta) - (link1*cos(q1) + link2*cos(q1 + q2))*sin(theta), -link2*sin(theta)*cos(q1 + q2) + link2*sin(q1 + q2)*cos(theta), 0, 0], 
        [0, 0, (-link1*sin(q3) - link2*sin(q3 + q4))*sin(theta) - (link1*cos(q3) + link2*cos(q3 + q4))*cos(theta), -link2*sin(theta)*sin(q3 + q4) - link2*cos(theta)*cos(q3 + q4)], 
        [0, 0, (-link1*sin(q3) - link2*sin(q3 + q4))*cos(theta) + (link1*cos(q3) + link2*cos(q3 + q4))*sin(theta), link2*sin(theta)*cos(q3 + q4) - link2*sin(q3 + q4)*cos(theta)]])

def elbowUpIK(X, Y, l1, l2):
    d = sqrt(X**2 + Y**2)
    calpha = (l1**2 + l2**2 - d**2) / (2 * l1 * l2)
    salpha = sqrt(1 - calpha**2)
    alpha = arctan2(salpha, calpha)
    q2 = pi - alpha
    alp = arctan2(Y, X)
    beta = arctan2(l2 * sin(q2), l1 + l2 * cos(q2))
    q1 = alp - beta
    return q1, q2

def elbowDownIK(X, Y, l1, l2):
    d = sqrt(X**2 + Y**2)
    calpha = (l1**2 + l2**2 - d**2) / (2 * l1 * l2)
    salpha = sqrt(1 - calpha**2)
    alpha = arctan2(salpha, calpha)
    q2 = pi - alpha
    alp = arctan2(Y, X)
    calpha1 = (l1**2 + d**2 - l2**2) / (2 * l1 * d)
    salpha1 = sqrt(1 - calpha1**2)
    beta = arctan2(salpha1, calpha1)
    q1 = alp + beta
    q2 = -q2
    return q1, q2

def trajectoryPlanner(to, tf, thetai, thetadi, thetaf, thetadf):
    Q = Matrix([thetai, thetadi, thetaf, thetadf])
    t0 = to
    B = Matrix([[1, t0, t0**2, t0**3],
                [0, 1, 2*t0, 3*t0**2],
                [1, tf, tf**2, tf**3],
                [0, 1, 2*tf, 3*tf**2]])
    Binv1 = B.inv()
    A1 = Binv1 * Q
    a0 = A1[0]
    a1 = A1[1]
    a2 = A1[2]
    a3 = A1[3]
    t = symbols('t')
    theta_d = a0 + a1*t + a2*t**2 + a3*t**3
    thetadot_d = a1 + 2*a2*t + 3*a3*t**2
    thetaddot_d = 2*a2 + 6*a3*t
    return theta_d, thetadot_d, thetaddot_d

'''
EKF 
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def rotZ(theta):
    return np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

def serialChainJacobian(l, phi):
    phi01 = phi[0] + phi[1] 
    return np.array([
        [(-l[0]*sin(phi[0]) - l[1]*sin(phi01)), -l[1]*sin(phi01)],
        [(l[0]*cos(phi[0]) + l[1]*cos(phi01)), l[1]*cos(phi01)]
    ])

def forwardKinematics(l, phi1, phi2):
    phi01 = phi1 + phi2
    x = l[0] * cos(phi1) + l[1] * cos(phi01)
    y = l[0] * sin(phi1) + l[1] * sin(phi01)
    return x, y

def stateTransitionFcn(Y, u, l1, l2, phi, shi):
    # Joint angles of the robot
    phi1 = phi[0]
    phi2 = phi[1]
    # Angle between palm frame and finger base frame 
    shi1 = shi[0]
    shi2 = shi[1]
    Rpk1 = rotZ(shi1)
    Rpk2 = rotZ(shi2)
    J1 = serialChainJacobian(l1, phi1)
    J2 = serialChainJacobian(l2, phi2)
    Y_pred = Y + np.vstack((np.hstack((Rpk1 @ J1, np.zeros(2,2))), np.hstack((np.zeros(2,2), Rpk2 @ J2)))) @ (u) 
    return Y_pred

def measurementFcn(l1, l2, phi):
    phi1 = phi[0]
    phi2 = phi[1]
    x1, y1 = forwardKinematics(l1, phi1)
    x2, y2 = forwardKinematics(l2, phi2)
    return np.array([[x1], [y1], [x2], [y2]])

''' 
Measurement/Sensor/Observation Model:

    Vector of predicted sensor measurements y at time t as a function of 
    the state x of the system at time t, plus some sensor noise w.

    y_t = H x_t + w_t

    The measurement matrix H is used to convert the predicted state estimate 
    at time t into predicted sensor measurements at time t.

    In this case, state 'x' comprises of [x, y, theta] therefore, H will be 
    identity.

    ArUco noise(error) for our sensor 
'''
def cameraMeasurement():
    


def EKFPredict(Y, P, Q, u, l1, l2, phi, shi):
    phi1 = phi[0]
    phi2 = phi[1]
    phi3 = phi[2]
    phi4 = phi[3]

    x1, y1 = forwardKinematics(L, phi1, phi2)
    x2, y2 = forwardKinematics(L, phi3, phi4)

    x = x1+x2/2
    y = y1+y2/2
    theta = arctan2((x1-x2)/(y1-y2))

    Y_pred = np.array([[x], [y], [theta]])
    F = np.eye(3)               # df is linear wrt dy and therefore, df/dy = eye(3)
    P_pred = F @ P @ F.T + Q    # Q is the covariance of the process noise 
    
    return Y_pred, P_pred       # A-priori state estimate and error covariance calculated using system model




    Y_pred = stateTransitionFcn(Y, u, l1, l2, phi, shi)
    # df is linear wrt dy and therefore, df/dy = eye(4)
    F = np.eye(4)
    # Q is the covariance of the process noise 
    P_pred = F @ P @ F.T + Q
    # A-priori state estimate and error covariance calculated using system model 
    return Y_pred, P_pred

def EKFUpdate(Y_pred, P_pred, R, z, l1, l2, phi):
    # Note that multiple measurements can be added along with their covariance and sampling rate
    z_pred = measurementFcn(l1, l2, phi)
    H = np.eye(4)
    S = H @ P_pred @ H.T + R  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    Y_updated = Y_pred + K @ (z - z_pred)
    # Where z is the actual measurement
    P_updated = (np.eye(len(K)) - K @ H) @ P_pred
    return Y_updated, P_updated

'''
DATA HANDLING AND LOGGING
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
latest_data = {'object_position': None}
    
def aruco_callback(data):
    position = data.data 
    update_dictionary('object_position', position)

def update_dictionary(key, value):
    latest_data[key] = value

'''
INITIALIZATIONS_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
# Define motor IDs
robotIDs = [11, 13, 21, 23]
redundantIDs = [12, 14, 22, 24]
robot = MultimotorControl(IDs = robotIDs)
redundantRobot = MultimotorControl(IDs = redundantIDs)

#Activate the torque 
robot.setMotor()
redundantRobot.setMotor()

positionMode = 3
currentMode = 0

dt = 0.001
to = 0
tf = 1
a = 0.0375 
Kt1 = 0.35
Kt2 = 0.51  

L = [0.103, 0.093]

# Initial End-Effector Positions______________________________________________________
x1, y1 = -80, 142
x2, y2 = 80, 135

q1, q2 = elbowUpIK(x1, y1, L[0]*1000, L[1]*1000)
q3, q4 = elbowDownIK(x2, y2, L[0]*1000, L[1]*1000)
homePosition = np.array([q1, q2, q3, q4]) + np.array([np.pi/2, np.pi, np.pi/2, np.pi])

nullForce = 500

timer = np.arange(to, tf + dt, dt)
trials = 100

lambda_ILC = 0.9
gammaX = 0
'''
overshot at 15000
overshot in 4th iteration at 5000
overshot in 9th iteration at 3000
'''
gammaY = 2000
gammaTheta = 0
gamma_ILC = np.array([[gammaX, 0, 0], [0, gammaY, 0], [0, 0, gammaTheta]])

# For impedence control
KpX = 5000
KpY = 0
KpTheta = 150
Kp = np.array([[KpX, 0, 0], [0, KpY, 0], [0, 0, KpTheta]])  

xDes = 0
thetaDes = 0

t = symbols('t')

xd, xdotd, xddotd = trajectoryPlanner(to, tf, 0, 0, 0, 0)
yd, ydotd, yddotd = trajectoryPlanner(to, tf, 0.139, 0, 0.101, 0)
th, thd, thdd = trajectoryPlanner(to, tf, 0, 0, 0, 0)

x = np.array([float(xd.subs(t, time)) for time in timer])
y = np.array( [float(yd.subs(t, time)) for time in timer])
theta = np.array([float(th.subs(t, time)) for time in timer])

desX = np.vstack((x, y, theta))
desX_trimmed = desX[:, 1:]
bodyF = [np.zeros((3, len(timer) - 1)) for _ in range(trials+1)]
actualPosition = [np.zeros((len(timer) - 1,3)) for _ in range(trials)]
G = [[None] * (len(timer) - 1) for _ in range(trials)]
fingerF = [[np.zeros(4)] * (len(timer) - 1) for _ in range(trials)]
tau = [[None] * (len(timer) - 1) for _ in range(trials)]
error = [[None] for _ in range(trials)]

def shutdown():
    robot.resetMotor()
    redundantRobot.resetMotor()
    rospy.loginfo("Hehe!")

# EKF Initializations
Y0 = np.array([0.0, 0.0, 0.0, 0.0])  # [x1, y1, x2, y2]
P0 = np.diag([0.1, 0.1, 0.1, 0.1])  # Covariance for [x1, y1, x2, y2]

l1 = l2 = [0.103, 0.93]
shi = 0

# Subscribe to joint angle publisher to fetch joint angles 'phi: phi' and joint velocities 'u: phid'


# Process noise covariance matrix 
Q = np.diag([0.01, 0.01, 0.01, 0.01])  # Process noise covariance for [x1, y1, x2, y2]
# Measurement noise covariance matrix (example, diagonal matrix)
R = np.diag([0.1, 0.1, 0.1, 0.1])  # Measurement noise covariance for [x1, y1, x2, y2]

# Increase Q if the predicted state Y_pred significantly diverges from the actual measurements, indicating underestimation of process noise.
# Increase R if the EKF estimates do not track sudden changes or measurements well, suggesting underestimation of measurement noise.
# Decrease Q or R if the filter is overly sensitive to noise or exhibits excessive variability in state estimates.

'''
ILC LOOP
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def ILC():
    
    for i in range(trials):

        redundantRobot.mode(positionMode)
        redundantRobot.sendPose([np.pi, np.pi, np.pi, np.pi])
        robot.mode(positionMode)
        robot.setMotor()
        
        robot.sendPose(homePosition)
        time.sleep(5)
        robot.resetMotor()

        robot.mode(currentMode)
        robot.setMotor()

        for j in range(len(timer) - 1):
           
            Q = latest_data['object_position']
            G[i][j] = grasp(Q[2], a)

            resultantF = bodyF[i]
            Fx = resultantF[0, j]
            Fy = resultantF[1, j]
            Mz = resultantF[2, j]

            redundantRobot.sendPose([np.pi, np.pi, np.pi, np.pi])

            actual_positions = robot.readPose()
            

            Jh = handJacobian(Q[2], L, actual_positions)
        

            f_null =  (np.identity(4) - (np.linalg.pinv(G[i][j]) @ G[i][j])) @ np.array([[nullForce],[nullForce],[nullForce],[nullForce]])  

            # f = np.dot(np.linalg.pinv(np.array(G[i][j], dtype=float)), np.array([[0], [Fy], [0]]))
            # f_impedence = np.dot(np.linalg.pinv(np.array(G[i][j], dtype=float)), np.dot(Kp,(np.array([[xDes- Q[0]], [0], [thetaDes - Q[2]]]))) ) 

            fingerF[i][j] = f_null

            JhT = np.transpose(Jh)
            tau[i][j] = np.dot(JhT,fingerF[i][j])
            goalCurrent = tau[i][j] 
            goalCurrent[0] = goalCurrent[0]/Kt1
            goalCurrent[1] = goalCurrent[1]/Kt2
            goalCurrent[2] = goalCurrent[2]/Kt1
            goalCurrent[3] = goalCurrent[3]/Kt2
        

            print(f"i: {i}, j: {j}, x: {Q[0]}, y: {Q[1]}, yaw: {Q[2]}, \
                current: {goalCurrent}, f_null: {f_null}")
            

            robot.sendCurrent(goalCurrent)
            actualPosition[i][j] = Q
            print(Q)

        actPosition = np.transpose(np.array(actualPosition[i]))
        error[i] = desX_trimmed - actPosition
        bodyF[i+1] = lambda_ILC * bodyF[i] + np.dot(gamma_ILC, error[i])
        robot.resetMotor() 

        print(f"actual position:{actPosition}, desired position: {desX_trimmed}")
        print(f"error: {error[i]}, body force {bodyF[i+1]}") 

'''
MAIN FUNCTION
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def main(): 
    rospy.init_node('ilc_node')

    
    rospy.loginfo("Waiting for ArUco data")

    rospy.Subscriber('aruco_data', Float64MultiArray, aruco_callback)

    while latest_data['object_position'] is None:
        rospy.sleep(0.1)

    rospy.loginfo("Fetched ArUco data") 

    ILC()

    if rospy.is_shutdown():
        rospy.loginfo("ROS node is shutting down.")
        return

if __name__ == '__main__':
    rospy.on_shutdown(shutdown) 
    main()