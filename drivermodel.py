import numpy as np
import vehiclemodel3dof
import matplotlib.pyplot as plt



vehicle = vehiclemodel3dof.VehicleDynamicModel3dof("mazda.ini")


target_speed = 170
speed_hist = 5

dt = 0.1
time = np.linspace(0, 100, 1001)
speed = list()
throttle = list()
error = list()

kp = 10
ki = 1
kd = 1

for t in time:

    """if vehicle.vehicle_speed_kmph < target_speed - speed_hist:
        vehicle.throttle = 100
    else:
        vehicle.throttle = 0"""

    e = target_speed - vehicle.vehicle_speed_kmph

    prop = kp * e
    integ = ki * e * dt
    der = kd * e / dt

    output = prop + integ + der
    vehicle.throttle = np.clip(output, 0, 100)

    vehicle.update(0.1)
    speed.append(vehicle.vehicle_speed_kmph)
    throttle.append(vehicle.throttle)
    error.append(e)


plt.plot(time, speed)
plt.plot(time, throttle)
plt.plot(time, error)

plt.show()
