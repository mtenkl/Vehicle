import configparser
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class VehicleDynamicModel3dof():

    def __init__(self, params=None) -> None:

        config = configparser.ConfigParser()

        if params is None:
            logging.warning("No parameter file. Using default values.")
        else:
            config.read(params)
        
        # Vehicle parameters
        self.wheel_base = config.getfloat("vehicle", "wheelbase", fallback=2.5)
        self.vehicle_mass = config.getfloat("vehicle", "mass", fallback=1390)
        self.vehicle_front_area = config.getfloat("vehicle", "frontArea", fallback=2.4)
        self.drag_coef = config.getfloat("vehicle", "dragCoeficient", fallback=0.3)

        # Engine parameters
        self.engine_speed_min = config.getfloat("engine", "engineSpeedMin", fallback=900)
        self.engine_speed_max = config.getfloat("engine", "engineSpeedMax", fallback=6000)
        self.max_torque_speed = config.getfloat("engine", "maxTorqueEngineSpeed", fallback=3500)
        self.max_power_speed = config.getfloat("engine", "maxPowerEngineSpeed", fallback=6000)
        self.torque_chart = list(map(float, config.get("engine", "torqueAxis", fallback="306 385 439 450 450 367").split()))
        self.torque_speed = list(map(float, config.get("engine", "engineSpeedAxis", fallback="900 2020 2990 3500 5000 6500").split()))
        self.power_chart = np.multiply(self.torque_chart, self.torque_speed) * np.pi / 30

        # Steering parameters
        self.steering_ratio = config.getfloat("steering", "steeringRatio", fallback=14)
        self.wheel_angle_max = config.getfloat("steering", "maxWheelAngle", fallback=32)

        # Transmission parameters
        self.gears_number = config.getfloat("transmission", "gears", fallback=6)
        self.gear_ratios = list(map(float, config.get("transmission", "gearRatios", fallback="4.71 3.14 2.11 1.67 1.29 1.00").split()))
        self.final_drive_ratio = config.getfloat("transmission", "finalDriveRatio", fallback=3)
        self.driveline_efficiency = config.getfloat("transmission", "driveLineEfficiency", fallback=0.9)

        # Tire parameters
        self.wheel_radius = config.getfloat("tire", "wheelRadius", fallback=0.9)
        self.dynamic_wheel_radius = self.wheel_radius * 0.98

        # Environment parameters
        self.air_density = config.getfloat("environment", "airDensity", fallback=1.225)
        self.road_load_coef = config.getfloat("environment", "roadLoadCoeficient", fallback=1.225)

        self.x = 0
        self.y = 0
        self.theta = 0
        self.wheel_angle = 0
        self.vehicle_speed = 0
        self.engine_speed = 1000
        self.slope = 0

    @property
    def vehicle_speed_kmph(self):
        return self.vehicle_speed * 3.6


    def set_position(self, x, y, theta, wheel_angle) -> None:

        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_angle = wheel_angle
        self.vehicle_speed = 0
    

    def steering(self, speed, wheel_angle_speed, dt):

        self.wheel_angle = self.wheel_angle + wheel_angle_speed * dt

        if self.wheel_angle > 0:
            self.wheel_angle = min(self.wheel_angle, self.wheel_angle_max)
        else:
            self.wheel_angle = max(self.wheel_angle, -self.wheel_angle_max)

        dx = speed * math.cos(self.theta)
        dy = speed * math.sin(self.theta)
        dtheta = speed * math.tan(self.wheel_angle) / self.wheel_base

        self.x = self.x + dx * dt
        self.y = self.y + dy * dt
        self.theta = self.theta + dtheta * dt

        return self.x, self.y, self.theta


    def driving_input(self, engine_torque: float, gear: str) -> None:
        """Driver`s inputs for vehicle model."""
        self.engine_torque = engine_torque
        self.selected_gear = gear


    def engine(self, engine_speed: float) -> tuple[float, float]:

        speed = np.clip(engine_speed, self.engine_speed_min, self.engine_speed_max)
        torque = np.interp(speed, self.torque_speed, self.torque_chart)
        return speed, torque

    def update(self, dt):

        self.engine_torque = self.engine(self.engine_speed)[1]

        traction_force = self.engine_torque * \
            self.gear_ratios[self.selected_gear] * self.final_drive_ratio * \
            self.driveline_efficiency / self.dynamic_wheel_radius

        road_slope_force = self.vehicle_mass * 9.81 * math.sin(self.slope)
        road_load_force = self.vehicle_mass * 9.81 * self.road_load_coef * math.cos(self.slope)
        aero_drag_force = 0.5 * self.air_density * self.drag_coef * self.vehicle_front_area * self.vehicle_speed * self.vehicle_speed

        self.vehicle_acceleration = (traction_force - road_slope_force - road_load_force - aero_drag_force) / self.vehicle_mass

        self.vehicle_speed = self.vehicle_speed + self.vehicle_acceleration * dt

def show_torque(vehicle: VehicleDynamicModel3dof):

    plt.subplot(1,2,1)
    plt.plot(vehicle.torque_speed, vehicle.torque_chart)
    plt.xlabel("Engine speed")
    plt.ylabel("Torque")

    plt.subplot(1,2,2)
    plt.plot(vehicle.torque_speed, vehicle.power_chart)
    plt.xlabel("Engine speed")
    plt.ylabel("Power")
    plt.show()

def main():

    vehicle = VehicleDynamicModel3dof("mazda.ini")
    vehicle.driving_input(100, 3)

    show_torque(vehicle)

    t = np.linspace(0,100, 1001)
    v = list()

    for tx in t:
        vehicle.update(0.1)
        v.append(vehicle.vehicle_speed_kmph)

    plt.plot(t, v)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.show()

    x = list()
    y = list()

    """
    for t in range(0, 800):

        wheel_angle_speed_deg = 1

        _x, _y, theta = vehicle.steering(
            2, wheel_angle_speed_deg/180*math.pi, 0.1)
        x.append(_x)
        y.append(_y)
    

    plt.plot(x, y)
    plt.show()

    """




if __name__ == "__main__":

    main()
