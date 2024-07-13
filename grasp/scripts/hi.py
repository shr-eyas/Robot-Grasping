import numpy as np

from numpy import sqrt, pi, cos, sin, arctan2, array 

# Time and Trial 
dt = 0.001
to = 0
tf = 1
timer = np.arange(to, tf + dt, dt)
trials = 100

bodyF = [np.zeros((3, len(timer) - 1)) for _ in range(trials+1)]

bodyF[0][0, :] = 10


print(bodyF[0][0, 3])  