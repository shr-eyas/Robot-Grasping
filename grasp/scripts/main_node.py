import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import rospy
import csv
import numpy as np
from numpy import sqrt, pi, cos, sin, arctan2, array 
from sympy import symbols, Matrix   
from std_msgs.msg import Float64MultiArray
from motorControl import MultimotorControl

if not os.path.exists('data'):
    os.makedirs('data')

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
DATA HANDLING AND LOGGING
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
latest_data = {'object_position': None}
 
def aruco_callback(data):
    position = data.data 
    update_dictionary('object_position', position)

def update_dictionary(key, value):
    latest_data[key] = value

def save_data_to_csv(file_name, data):
    file_path = os.path.join('data', file_name)

    if data:
        header = ['i', 'j'] + list(data[0].keys())[2:]  
    else:
        return 

    rows = []
    for entry in data:
        row = [entry['i'], entry['j']] + list(entry.values())[2:]
        rows.append(row)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)



data_storage = {'actualPosition': [], 'f_learnt': [], 'goalCurrent': [], 'bodyForces': []}


'''
INITIALIZATIONS_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''

# For Dynamixel Motors
robotIDs =      [11, 13, 21, 23]
redundantIDs =  [12, 14, 22, 24]
robot =             MultimotorControl(IDs = robotIDs)
redundantRobot =    MultimotorControl(IDs = redundantIDs)
robot.setMotor()
redundantRobot.setMotor()
positionMode =  3
currentMode =   0
Kt1 = 0.35
Kt2 = 0.51 

# Hardware Related Initializations
L = [0.104, 0.084]
a = 0.027 
x1, y1 = -83, 111
x2, y2 = 81, 108
q1, q2 = elbowUpIK(x1, y1, L[0]*1000, L[1]*1000)
q3, q4 = elbowDownIK(x2, y2, L[0]*1000, L[1]*1000)
homePosition = np.array([q1, q2, q3, q4]) + np.array([np.pi/2, np.pi, np.pi/2, np.pi])

# Time and Trial 
dt = 0.001
to = 0
tf = 1
timer = np.arange(to, tf + dt, dt)
trials = 100

# For ILC
lambda_ILC = 0.9    # Retention Rate 
gammaX = 0          # Learning Rate along X
gammaY = 15000       # Learning Rate along Y
gammaTheta = 0      # Learning Rate along Yaw
# overshot at 15000
# overshot in 4th iteration at 5000
# overshot in 9th iteration at 3000
gamma_ILC = np.array([[gammaX, 0, 0], 
                      [0, gammaY, 0], 
                      [0, 0, gammaTheta]])

# For Impedence Control
KpX = 40000               # Impedance along X
KpY = 0               # No Impedance along Y; 15000
KpTheta = 45           # Impedance along Yaw
impedanceK = np.array([[KpX, 0, 0], 
                       [0, KpY, 0], 
                       [0, 0, KpTheta]])  
setPointX = 0           # Setpoint for Impedance along X
setPointY = 0.14217     # Setpoint for Impedance along Y
setPointTheta = 0       # Setpoint for Impedance along Yaw

# For Finger Forces 
nullForce = 600
nullVector = np.array([[nullForce],
                       [nullForce],
                       [nullForce],
                       [nullForce]])
W = 1 # Learning 
I = 1 # Impedance
N = 1 # Null 

t = symbols('t')

xd, xdotd, xddotd = trajectoryPlanner(to, tf, 0, 0, 0, 0)
yd, ydotd, yddotd = trajectoryPlanner(to, tf, 0.1082, 0, 0.1382, 0)
th, thd, thdd =     trajectoryPlanner(to, tf, 0, 0, 0, 0)

X = np.array([float(xd.subs(t, time)) for time in timer])
Y = np.array( [float(yd.subs(t, time)) for time in timer])
Theta = np.array([float(th.subs(t, time)) for time in timer])

desX = np.vstack((X, Y, Theta))
desX_trimmed = desX[:, 1:]
bodyF = [np.zeros((3, len(timer) - 1)) for _ in range(trials+1)]
actualPosition = [np.zeros((len(timer) - 1,3)) for _ in range(trials)]
G = [[None] * (len(timer) - 1) for _ in range(trials)]
fingerF = [[np.zeros(4)] * (len(timer) - 1) for _ in range(trials)]
tau = [[None] * (len(timer) - 1) for _ in range(trials)]
error = [[None] for _ in range(trials)]

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
        rospy.sleep(3)
        robot.resetMotor()

        robot.mode(currentMode)
        robot.setMotor()

        for j in range(len(timer) - 1):

            # Lock Joints 
            redundantRobot.sendPose([np.pi, np.pi, np.pi, np.pi])

            # Data Updates
            objectPosition = latest_data['object_position']
            jointPositions = robot.readPose()

            # Calculate Grasp Matrix
            G[i][j] = grasp(a, objectPosition[2])      

            # Body Wrench
            resultantF = bodyF[i] 
            w = np.array([[resultantF[0, j]],           
                          [resultantF[1, j]], 
                          [resultantF[2, j]]])
            w_imp_k = impedanceK @ np.array([[setPointX - objectPosition[0]], 
                                           [setPointY - objectPosition[1]], 
                                           [setPointTheta - objectPosition[2]]]) 
            w_imp_d = 0 
            w_imp = w_imp_k + w_imp_d
            
            # Hand Jacobian
            Jh = handJacobian(objectPosition[2], L, jointPositions)
            JhT = np.transpose(Jh)

            # Finger Force Components 
            f_null =  (np.identity(4) - (np.linalg.pinv(G[i][j]) @ G[i][j])) @ nullVector  
            f_learnt = np.linalg.pinv(G[i][j]) @ w 
            f_impedence = np.linalg.pinv(G[i][j]) @ w_imp 
            fingerF[i][j] = W*f_learnt + I*f_impedence + N*f_null  

            # Torques and Current 
            tau[i][j] = JhT @ fingerF[i][j]
            goalCurrent = tau[i][j] 
            goalCurrent[0] = goalCurrent[0]/Kt1
            goalCurrent[1] = goalCurrent[1]/Kt2
            goalCurrent[2] = goalCurrent[2]/Kt1
            goalCurrent[3] = goalCurrent[3]/Kt2

            # Set Current
            robot.sendCurrent(goalCurrent)

            # Data Logging 
            actualPosition[i][j] = objectPosition

            data_storage['actualPosition'].append({'i': i, 'j': j, 'x': objectPosition[0], 'y': objectPosition[1], 'yaw': objectPosition[2]})
            data_storage['f_learnt'].append({'i': i, 'j': j, 'f_learnt': f_learnt.flatten().tolist()})
            data_storage['goalCurrent'].append({'i': i, 'j': j, 'goalCurrent': goalCurrent.flatten().tolist()})
            data_storage['bodyForces'].append({'i': i, 'j': j, 'bodyF': resultantF[:, j].flatten().tolist()})


            print(objectPosition)
            print(f"i: {i}, j: {j}, \n x: {objectPosition[0]}, y: {objectPosition[1]}, yaw: {objectPosition[2]}, \
                \n current: {goalCurrent}, \n f_learnt: {W*f_learnt}, \n f_impedance: {I*f_impedence},\n f_null: {N*f_null}")
  
        actPosition = np.transpose(np.array(actualPosition[i]))
        error[i] = desX_trimmed - actPosition
        bodyF[i+1] = lambda_ILC * bodyF[i] + (gamma_ILC @ error[i])
        robot.resetMotor() 

        print(f"actual position:{actPosition}, desired position: {desX_trimmed}")
        print(f"error: {error[i]}, body force {bodyF[i+1]}") 

        save_data_to_csv('actualPosition.csv', data_storage['actualPosition'])
        save_data_to_csv('f_learnt.csv', data_storage['f_learnt'])
        save_data_to_csv('goalCurrent.csv', data_storage['goalCurrent'])
        save_data_to_csv('bodyForces.csv', data_storage['bodyForces'])

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
    rospy.loginfo("Executing ILC Loop") 
    rospy.sleep(2)

    ILC()

    if rospy.is_shutdown():
        rospy.loginfo("ROS node is shutting down.")
        return

if __name__ == '__main__':
    main()