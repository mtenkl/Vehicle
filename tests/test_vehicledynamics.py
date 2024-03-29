import math
import matplotlib.pyplot as plt
import numpy as np
from vehicle import vehicledynamics
import pytest


vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")


def test_steering():

    vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")

    traj = np.zeros((100, 3))
    steering_speed_rad = -30 /180 * math.pi
    theta_x = 0 /180 * math.pi
    vehicle.set_position(theta=theta_x)
    vehicle_speed = 0.5

    for i in range(100):
        
        x, y, theta = vehicle.steering(vehicle_speed, steering_speed_rad, 0.1)
        traj[i][0] = x
        traj[i][1] = y
        traj[i][2] = theta / math.pi * 180

    plt.subplot(2,2,1)
    plt.title("x")
    plt.plot(traj[:,0])
    
    plt.subplot(2,2,2)
    plt.title("y")
    plt.plot(traj[:,1])
    
    plt.subplot(2,2,3)
    plt.title("theta deg")
    plt.plot(traj[:,2])

    plt.subplot(2,2,4)
    plt.title("[x,y]")
    plt.plot(traj[:, 0], traj[:,1], "-")

    plt.tight_layout()
    plt.show()


def test_accelerating():
    vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")
    vehicle.ignition_on = True
    vehicle.drive_mode = "D"

    vehicle.throttle_pedal = 40
    vehicle.update(10)

    assert vehicle.vehicle_speed_kmph > 100
    assert vehicle.vehicle_speed_kmph < 180


def test_neutral_gear():
    vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")
    vehicle.ignition_on = True
    vehicle.drive_mode = "N"

    vehicle.throttle_pedal = 40
    for i in range(1000):
        vehicle.update(0.01)

    assert vehicle.vehicle_speed_kmph == 0

def test_braking():
    vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")
    vehicle.ignition_on = True
    vehicle.drive_mode = "D"
    vehicle.throttle_pedal = 40
    vehicle.update(10)

    assert vehicle.vehicle_speed_kmph > 100
    assert vehicle.vehicle_speed_kmph < 180

    vehicle.brake_pedal = 90
    vehicle.throttle_pedal = 0
    vehicle.update(4)
    assert vehicle.vehicle_speed_kmph > 10
    assert vehicle.vehicle_speed_kmph < 60


def test_braking_to_full_stop():

    vehicle = vehicledynamics.VehicleDynamicModel3dof("vehicle/mazda.ini")
    vehicle.ignition_on = True
    vehicle.drive_mode = "D"

    vehicle.throttle_pedal = 0
    #vehicle.update(2)
    #assert vehicle.vehicle_speed_kmph > 5, "Speed should be above 5"
    #assert vehicle.vehicle_speed_kmph < 15

    vehicle.brake_pedal = 90
    vehicle.throttle_pedal = 0

    speeds = list()
    slope = list()
    drag = list()
    braking = list()
    long = list()
    trac = list()
    roll = list()

    i = 1
    while i < 1000: # vehicle.vehicle_speed_kmph > 0.1:


        if i > 0:
            vehicle.drive_mode = "D"
            vehicle.throttle_pedal = 20
            vehicle.brake_pedal = 0
        if i > 300:
            vehicle.drive_mode = "D"
            vehicle.throttle_pedal = 0
            vehicle.brake_pedal = 80
        if i > 500:
            vehicle.drive_mode = "D"
            vehicle.throttle_pedal = 0
            vehicle.brake_pedal = 0
        if i > 700:
            vehicle.drive_mode = "R"
            vehicle.throttle_pedal = 0
            vehicle.brake_pedal = 80
        if i > 800:
            vehicle.drive_mode = "R"
            vehicle.throttle_pedal = 20
            vehicle.brake_pedal = 0


        vehicle.update(0.01)
        i += 1
        speeds.append(vehicle.vehicle_speed_kmph)
        slope.append(vehicle._slope_force)
        drag.append(vehicle._drag_force)
        braking.append(vehicle._braking_force)
        long.append(vehicle._longitudinal_force)
        trac.append(vehicle._traction_force)
        roll.append(vehicle._rolling_force)

    plt.subplot(2,1,1)
    plt.plot(speeds)
    plt.subplot(2,1,2)
    plt.plot(slope)
    plt.plot(drag)
    plt.plot(braking)
    plt.plot(long)
    plt.plot(trac)
    plt.plot(roll)
    plt.legend(["slope", "drag", "braking", "long", "traction", "rolling"])
    plt.show()
    return
    assert vehicle.vehicle_speed_kmph > 0
    assert vehicle.vehicle_speed_kmph < 10