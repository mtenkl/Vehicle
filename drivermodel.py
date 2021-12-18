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
target_v = list()
traction = list()


kp = 10
ki = 1
kd = 1

for t in time:

    """if vehicle.vehicle_speed_kmph < target_speed - speed_hist:
        vehicle.throttle = 100
    else:
        vehicle.throttle = 0"""

    if t > 20:
        target_speed = 100
    if t > 40:
        target_speed = 50
    if t > 60:
        target_speed = 130

    e = target_speed - vehicle.vehicle_speed_kmph

    prop = kp * e
    integ = ki * e * dt
    der = kd * e / dt

    output = prop + integ + der
    output = 100
    if t > 20:
        output = 0

    vehicle.throttle = np.clip(output, 0, 100)

    vehicle.update(0.1)
    speed.append(vehicle.vehicle_speed_kmph)
    throttle.append(vehicle.throttle)
    traction.append(vehicle._total_force)
    error.append(e)
    target_v.append(target_speed)

plt.subplot(2, 3, 1)
plt.plot(time, speed)
plt.subplot(2, 3, 2)
plt.plot(time, throttle)
plt.subplot(2, 3, 3)
plt.plot(time, target_v)
plt.subplot(2, 3, 4)
plt.plot(time, traction)
plt.legend()

plt.show()
