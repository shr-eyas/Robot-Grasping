import numpy as np
from dynamixel_client import DynamixelClient

class MultimotorControl():
    def __init__(self, IDs , baud_rate = 4000000):
        # Motor IDs (Mention All motor IDs) 
        self.motors = motors = IDs 
        
        try:
            print("Trying to connect to /dev/ttyUSB0")
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', baud_rate)
            self.dxl_client.connect()
            print("Connected to /dev/ttyUSB0")
        except Exception as e:
            print(f"Failed to connect to /dev/ttyUSB0: {e}")
            try:
                print("Trying to connect to /dev/ttyUSB1")
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', baud_rate)
                self.dxl_client.connect()
                print("Connected to /dev/ttyUSB1")
            except Exception as e:
                print(f"Failed to connect to /dev/ttyUSB1: {e}")
                try:
                    print("Trying to connect to /dev/ttyUSB2")
                    self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', baud_rate)
                    self.dxl_client.connect()
                    print("Connected to /dev/ttyUSB2")
                except Exception as e:
                    print(f"Failed to connect to /dev/ttyUSB2: {e}")
                    raise Exception("Could not connect to any of the specified ports")


        Kp = 1000
        Kd = 0
        Ki = 0
        
        ADDR_SET_MOTOR_KP = 84 
        LEN_SET_MOTOR_KP = 2
        self.dxl_client.sync_write(motors,np.ones(len(motors)) * Kp, ADDR_SET_MOTOR_KP,LEN_SET_MOTOR_KP)

        ADDR_SET_MOTOR_KD = 80
        LEN_SET_MOTOR_KD = 2
        self.dxl_client.sync_write(motors,np.ones(len(motors)) * Kd, ADDR_SET_MOTOR_KD,LEN_SET_MOTOR_KD)            

        ADDR_SET_MOTOR_KI = 82
        LEN_SET_MOTOR_KI = 2
        self.dxl_client.sync_write(motors,np.ones(len(motors)) * Ki, ADDR_SET_MOTOR_KI,LEN_SET_MOTOR_KI)            

    def mode(self, mode):
        ADDR_SET_MODE = 11
        LEN_SET_MODE = 1
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors))*mode, ADDR_SET_MODE, LEN_SET_MODE)

    def sendPose(self, position):
        self.curr_pos = np.array(position)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def sendCurrent(self, current):
        self.curr_cur = np.array(current)
        self.dxl_client.write_desired_cur(self.motors, self.curr_cur)

    def setMotor(self):
        self.dxl_client.set_torque_enabled(self.motors, True)

    def resetMotor(self):
        self.dxl_client.set_torque_enabled(self.motors, False)

    def readPose(self):     
        pos = self.dxl_client.read_pos() 
        pos[0] = pos[0] - np.pi/2
        pos[1] = pos[1] - np.pi
        pos[2] = pos[2] - np.pi/2
        pos[3] = pos[3] - np.pi
        return pos
    
    def read_positions(self):
        pos = self.dxl_client.read_pos()
        pos = np.rad2deg(pos)
        pos[0] = pos[0] - 90
        pos[1] = pos[1] - 180
        pos[2] = pos[2] - 90
        pos[3] = pos[3] - 180
        return pos
    
    def readVelocity(self):
        return self.dxl_client.read_vel()

    def readCurrent(self):
        return self.dxl_client.read_cur()