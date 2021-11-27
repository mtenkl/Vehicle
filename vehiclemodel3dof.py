import configparser
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class VehicleDynamicModel3dof():

    def __init__(self, params=None) -> None:

        

        if params is not None:
            logging.info("Loading {} parameters file.".format(params))
            
            config = configparser.ConfigParser()
            config.read(params)
            body = config["body"]
            powertrain = config["powertrain"]
            self.wheel_base = body.getfloat("wheelbase", 2.5)
            self.vehicle_mass = body.getfloat("mass", 1390)
            self.steering_ratio = body.getfloat("steeringratio", 14)
            self.wheel_angle_max = body.getfloat("maxwheelangle", 0.523)

            self.gear_ratios = self._parse_gear_ratios(powertrain, 0)
            self.final_drive_ratio = powertrain.getfloat("finaldrive", 3)
            self.driveline_efficiency = powertrain.getfloat(
                "drivelineefficiency", 0.9)
            self.wheel_radius = powertrain.getfloat("wheelradius", 0.33)
            self.air_density = config["environment"].getfloat("airdensity", 1.225)
            self.drag_coef = config["body"].getfloat("dragcoef", 0.3)
            self.vehicle_front_area = config["body"].getfloat("frontarea", 2.1)


        self.x = 0
        self.y = 0
        self.theta = 0
        self.wheel_angle = 0
        self.vehicle_speed = 0

        self.slope = 0

    @property
    def vehicle_speed_kmph(self):
        return self.vehicle_speed * 3.6

    def _parse_gear_ratios(self, params: dict, default: float) -> dict:
        """Parses gear ratios from configuration."""

        ratios = dict()
        for i in range(-1, 10):
            gear_name = "gear" + str(i)
            if gear_name in params:
                ratios[str(i)] = params.getfloat(gear_name, default)

        return ratios

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

    def update(self, dt):

        dynamic_wheel_radius = self.wheel_radius * 0.98
        traction_force = self.engine_torque * \
            self.gear_ratios[self.selected_gear] * self.final_drive_ratio * \
            self.driveline_efficiency / dynamic_wheel_radius

        road_slope_force = self.vehicle_mass * 9.81 * math.sin(self.slope)
        road_load_force = self.vehicle_mass * 9.81 * 0.01 * math.cos(self.slope)
        aero_drag_force = 0.5 * self.air_density * self.drag_coef * self.vehicle_front_area * self.vehicle_speed * self.vehicle_speed



        self.vehicle_acceleration = (traction_force - road_slope_force - road_load_force - aero_drag_force) / self.vehicle_mass

        self.vehicle_speed = self.vehicle_speed + self.vehicle_acceleration * dt



def main():

    vehicle = VehicleDynamicModel3dof("mazda.ini")
    vehicle.driving_input(100, "6")


    t = np.linspace(0,100, 1001)
    v = list()

    for tx in t:
        vehicle.update(0.1)
        v.append(vehicle.vehicle_speed_kmph)

    plt.plot(t, v)
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
