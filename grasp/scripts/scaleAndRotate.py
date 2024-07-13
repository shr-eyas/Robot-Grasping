import csv
import ast  
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, pi   


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

# Time and Trial 
dt = 0.001
to = 0
tf = 1
timer = np.arange(to, tf + dt, dt)
trials = 100


t = symbols('t')

xd, xdotd, xddotd = trajectoryPlanner(to, tf, 0, 0, 0, 0)
yd, ydotd, yddotd = trajectoryPlanner(to, tf, 0.1082, 0, 0.1382, 0)

xTrajectory = np.array([float(xd.subs(t, time)) for time in timer])
yTrajectory = np.array( [float(yd.subs(t, time)) for time in timer])

xStart = np.array([-0.03, 0.1313])
xGoal = np.array([0.037, 0.098])

transformed_positions, scaling, rotationAngle = scaleAndRotateTrajectory(xStart, xGoal, xTrajectory, yTrajectory)
newBodyForces = scaleAndRotateForces(scaling, rotationAngle, xForces, yForces)
print(transformed_positions[:,1])
print(f'The trajectory is scaled by {scaling} and rotated by {np.degrees(rotationAngle)} degrees.')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(xTrajectory, yTrajectory, label='Initial Trajectory', color='blue')
plt.plot(transformed_positions[:, 0], transformed_positions[:, 1], label='Transformed Trajectory', color='red')
plt.scatter([xStart[0], xGoal[0]], [xStart[1], xGoal[1]], color='green', zorder=5)  # Mark start and goal points
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Initial and Transformed Trajectories')
plt.grid(True)
plt.show()











