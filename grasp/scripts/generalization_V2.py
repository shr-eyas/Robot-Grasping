import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import rospy
import csv
import ast  
import numpy as np
from numpy import sqrt, pi, cos, sin, arctan2, array 
from sympy import symbols, Matrix   
from std_msgs.msg import Float64MultiArray
from motorControl import MultimotorControl
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

if not os.path.exists('data'):
    os.makedirs('data')

'''
FETCH GENERALIZATION DATA
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
bodyForcePath = '/home/sysidea/grasp_ws/src/data_30mm_40trials/bodyForces.csv'

bodyForceList = []

with open(bodyForcePath, mode='r', newline='') as file1:
    reader = csv.reader(file1)
    header = next(reader) 
    for row in reader:
        i = int(row[0])
        if i == 41:
            goalCurrent_str = row[2] 
            goalCurrent_ = ast.literal_eval(goalCurrent_str)          
            bodyForceList.append(goalCurrent_)

bodyForces = np.array(bodyForceList)
xForces = bodyForces[:,0]
yForces = bodyForces[:,1]

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

def scaleAndRotateTrajectory(xStart, xGoal, xTraj, yTraj):

    initialVector = np.array([xTraj[-1] - xTraj[0], yTraj[-1] - yTraj[0]])
    goalVector = np.array([xGoal[0] - xStart[0], xGoal[1] - xStart[1]])
    
    scaling = np.linalg.norm(goalVector) / np.linalg.norm(initialVector)
    
    initialAngle = np.arctan2(initialVector[1], initialVector[0])
    goalAngle = np.arctan2(goalVector[1], goalVector[0])
    
    rotationAngle = goalAngle - initialAngle
    
    rotationMatrix = np.array([[np.cos(rotationAngle), -np.sin(rotationAngle)],
                               [np.sin(rotationAngle), np.cos(rotationAngle)]])
    
    translatedPositions = np.vstack((xTraj - xTraj[0], yTraj - yTraj[0]))
    scaledPositions = scaling * rotationMatrix @ translatedPositions
    transformedPositions = scaledPositions + np.array([[xStart[0]], [xStart[1]]])
    
    return transformedPositions.T, scaling, rotationAngle

def scaleAndRotateForces(scaling, rotationAngle, xForces, yForces):
   
    rotationMatrix = np.array([[np.cos(rotationAngle), -np.sin(rotationAngle)],
                               [np.sin(rotationAngle), np.cos(rotationAngle)]])
    
    transformedForces = scaling * rotationMatrix @ np.vstack((xForces, yForces))
    
    return transformedForces.T

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
gammaX = 15000          # Learning Rate along X
gammaY = 5000       # Learning Rate along Y
gammaTheta = 0      # Learning Rate along Yaw
# overshot at 15000
# overshot in 4th iteration at 5000
# overshot in 9th iteration at 3000
gamma_ILC = np.array([[gammaX, 0, 0], 
                      [0, gammaY, 0], 
                      [0, 0, gammaTheta]])

# For Impedence Control
lambdaX = 0
lambdaY = 0
lambdaL = np.array([[lambdaX, 0], 
                    [0, lambdaY]])  
KpTheta = 50            # Impedance along Yaw
impedanceK = np.array([[0, 0, 0], 
                       [0, 0, 0], 
                       [0, 0, KpTheta]])  
setPointTheta = 0       # Setpoint for Impedance along Yaw

# For Finger Forces 
nullForce = 450
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


bodyF = [np.zeros((3, len(timer) - 1)) for _ in range(trials+1)]
actualPosition = [np.zeros((len(timer) - 1,3)) for _ in range(trials)]
G = [[None] * (len(timer) - 1) for _ in range(trials)]
fingerF = [[np.zeros(4)] * (len(timer) - 1) for _ in range(trials)]
tau = [[None] * (len(timer) - 1) for _ in range(trials)]
error = [[None] for _ in range(trials)]

# Scale and rotate the trajectory and the force
xStart = np.array([0, 0.1082])
xGoal = np.array([0.02, 0.135])

transformedPositions, scaling, rotationAngle = scaleAndRotateTrajectory(xStart, xGoal, X, Y)
newBodyForces = scaleAndRotateForces(scaling, rotationAngle, xForces, yForces)
desX = np.vstack((transformedPositions[:,0], transformedPositions[:,1], Theta))
desX_trimmed = desX[:, 1:]
# Construct the kd-tree
KDtree = KDTree(transformedPositions)


'''
GENERALIZATION LOOP
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def generalization():
    
    for i in range(trials):

        redundantRobot.mode(positionMode)
        redundantRobot.sendPose([np.pi, np.pi, np.pi, np.pi])
        robot.mode(positionMode)
        robot.setMotor()
        
        robot.sendPose(homePosition)
        rospy.sleep(2)
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
            bodyF[0][0, j] = newBodyForces[j, 0]
            bodyF[0][1, j] = newBodyForces[j, 1]

            resultantF = bodyF[i] 
            
            w = np.array([[resultantF[0, j]],      
                          [resultantF[1, j]],                                             
                          [resultantF[2, j]]])

            # Path Dependent Impedance
            positionXY = objectPosition[:2]
            _, i_min = KDtree.query(positionXY)
            v_fb = lambdaL @ (transformedPositions[i_min] - positionXY)
            w_imp = impedanceK @ np.array([[v_fb[0]], 
                                           [v_fb[1]], 
                                           [setPointTheta - objectPosition[2]]])  

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
            actualCurrent = robot.readCurrent()

            # Data Logging 
            actualPosition[i][j] = objectPosition

            data_storage['actualPosition'].append({'i': i, 'j': j, 'x': objectPosition[0], 'y': objectPosition[1], 'yaw': objectPosition[2]})
            data_storage['f_learnt'].append({'i': i, 'j': j, 'f_learnt': f_learnt.flatten().tolist()})
            data_storage['goalCurrent'].append({'i': i, 'j': j, 'goalCurrent': goalCurrent.flatten().tolist()})
            data_storage['bodyForces'].append({'i': i, 'j': j, 'bodyF': resultantF[:, j].flatten().tolist()})

            print(objectPosition)
            print(f"i: {i}, j: {j}, \n x: {objectPosition[0]}, y: {objectPosition[1]}, yaw: {objectPosition[2]}, \
                \n current: {goalCurrent}, \n actualCurrent: {actualCurrent} \n f_learnt: {W*f_learnt}, \n f_impedance: {I*f_impedence},\n f_null: {N*f_null}, \n Combined Forces: {fingerF[i][j]}")
            
        actPosition = np.transpose(np.array(actualPosition[i]))
        error[i] = desX_trimmed - actPosition
        bodyF[i+1] = lambda_ILC * bodyF[i] + (gamma_ILC @ error[i])
        robot.resetMotor() 

        save_data_to_csv('actualPosition.csv', data_storage['actualPosition'])
        save_data_to_csv('f_learnt.csv', data_storage['f_learnt'])
        save_data_to_csv('goalCurrent.csv', data_storage['goalCurrent'])
        save_data_to_csv('bodyForces.csv', data_storage['bodyForces'])


'''
MAIN FUNCTION
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
'''
def main(): 
    rospy.init_node('generalization_node')
    rospy.loginfo("Waiting for ArUco data")
    rospy.Subscriber('aruco_data', Float64MultiArray, aruco_callback)

    while latest_data['object_position'] is None:
        rospy.sleep(0.1)

    rospy.loginfo("Fetched ArUco data") 
    rospy.loginfo("Executing Generalization Loop") 
    rospy.sleep(2)

    generalization()

    if rospy.is_shutdown():
        rospy.loginfo("ROS node is shutting down.")
        return

if __name__ == '__main__':
    main()

