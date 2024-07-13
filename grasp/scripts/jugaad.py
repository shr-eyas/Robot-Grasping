import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import time

from motorControl import MultimotorControl


start_time = time.time()

# Define motor IDs
motor_ID = [11, 12, 13, 14, 21, 22, 23, 24]
Two_finger_robot = MultimotorControl(IDs = motor_ID)

# Set Operating Mode
Two_finger_robot.mode(3)

#Activate the torque 
Two_finger_robot.setMotor()



# [87.07029095976552, 69.39547040114685,  94.02390127831237, -73.73119684864211]
while True:
    goal_positions = np.radians(np.array([87.07029095976552,0, 69.39547040114685,0, 94.02390127831237, 0,-73.73119684864211,0]))
    offset = np.array([np.pi/2, np.pi, np.pi, np.pi, np.pi/2, np.pi, np.pi, np.pi])
    goal_positions_final = (goal_positions + offset)
    Two_finger_robot.sendPose(goal_positions_final)

    current_time = time.time()
    elapsed_time = current_time - start_time

    pos = Two_finger_robot.readPose()
    print(pos)

    # if elapsed_time > 10: 
    #     Two_finger_robot.torque_de_activate(motor_ID)
    #     break

    # time.sleep(0.03)














