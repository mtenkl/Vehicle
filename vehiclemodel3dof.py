import configparser
import argparse
import math
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class VehicleDynamicModel3dof():

    def __init__(self, params=None) -> None:

        self.simulation_step = None

        config = configparser.ConfigParser()

        if params is None:
            logging.warning("No parameter file. Using default values.")
        else:
            config.read(params)

        # Vehicle parameters
        self.wheel_base = config.getfloat("vehicle", "wheelbase", fallback=2.5)
        self.vehicle_mass = config.getfloat("vehicle", "mass", fallback=1390)
        self.vehicle_front_area = config.getfloat(
            "vehicle", "frontArea", fallback=2.4)
        self.vehicle_drag_coef = config.getfloat(
            "vehicle", "dragCoeficient", fallback=0.3)

        # Engine parameters
        self.engine_speed_min = config.getfloat(
            "engine", "engineSpeedMin", fallback=900)
        self.engine_speed_max = config.getfloat(
            "engine", "engineSpeedMax", fallback=6000)
        self.engine_max_torque_speed = config.getfloat(
            "engine", "maxTorqueEngineSpeed", fallback=3500)
        self.engine_max_power_speed = config.getfloat(
            "engine", "maxPowerEngineSpeed", fallback=6000)
        self.engine_torque_curve = list(map(float, config.get(
            "engine", "torqueAxis", fallback="306 385 439 450 450 367").split()))
        self.engine_speed_curve = list(map(float, config.get(
            "engine", "engineSpeedAxis", fallback="900 2020 2990 3500 5000 6500").split()))
        self.engine_power_curve = np.multiply(
            self.engine_torque_curve, self.engine_speed_curve) * np.pi / 30
        self.engine_braking_torque_curve = list(map(float, config.get(
            "engine", "brakingTorqueAxis", fallback="30 40 48 45 30 20").split()))

        # Steering parameters
        self.steering_ratio = config.getfloat(
            "steering", "steeringRatio", fallback=14)
        self.steering_wheel_angle_max = config.getfloat(
            "steering", "maxWheelAngle", fallback=32)

        # Transmission parameters
        self.transmission_gears_number = config.getint(
            "transmission", "gears", fallback=6)
        self.transmission_gear_ratios = list(map(float, config.get(
            "transmission", "gearRatios", fallback="4.71 3.14 2.11 1.67 1.29 1.00").split()))
        self.transmission_final_drive_ratio = config.getfloat(
            "transmission", "finalDriveRatio", fallback=3)
        self.transmission_driveline_efficiency = config.getfloat(
            "transmission", "driveLineEfficiency", fallback=0.9)

        # Tire parameters
        self.tire_wheel_radius = config.getfloat(
            "tire", "wheelRadius", fallback=0.9)
        self.tire_dynamic_wheel_radius = self.tire_wheel_radius * 0.98

        # Environment parameters
        self.env_air_density = config.getfloat(
            "environment", "airDensity", fallback=1.225)
        self.env_road_load_coef = config.getfloat(
            "environment", "roadLoadCoeficient", fallback=1.225)
        self.env_gravity_acceleration = config.getfloat(
            "environment", "gravityAcceleration", fallback=9.81)

        # Brakes parameters
        self.brakes_front_pads_count = config.getfloat(
            "brakes", "frontPads", fallback=2)
        self.brakes_rear_pads_count = config.getfloat(
            "brakes", "rearPads", fallback=2)
        self.brakes_static_friction = config.getfloat(
            "brakes", "staticFriction", fallback=0.92)
        self.brakes_dynamic_friction = config.getfloat(
            "brakes", "dynamicFriction", fallback=0.9)
        self.brake_piston_diameter = config.getfloat(
            "brake", "pistonDiameter", fallback=0.02)
        self.brake_pad_position_radius = config.getfloat(
            "brake", "padPositionRadius", fallback=0.15)
        self.brake_position_curve = list(map(float, config.get(
            "brake", "brakePositionPct", fallback="0 5 30 50 70 100").split()))
        self.brake_pressure_position_curve = list(map(float, config.get(
            "brake", "brakePositionPressure", fallback="0 0 15e5 30e5 50e5 60e5").split()))

        self.x = 0
        self.y = 0
        self.slope = 0

        # Vehicle variables
        self._vehicle_speed = 0
        self._vehicle_acceleration = 0
        self._wheel_speed = 0

        self._traction_force = 0
        self._longitudinal_force = 0

        # Engine variables
        self._engine_speed = 0
        self._engine_torque = 0
        self._throttle = 0

        # Transmission variables
        self._selected_gear = 1

        # Environment
        self._drag_force = 0
        self._rolling_force = 0
        self._slope_force = 0

        # Steering variables
        self.theta = 0
        self.wheel_angle = 0

        # Braking variables
        self._braking_torque = 0
        self._braking_force = 0
        self._brake_pedal = 0

    # Vehicle properties

    @property
    def vehicle_speed_mps(self):
        return self._vehicle_speed

    @property
    def vehicle_speed_kmph(self):
        return self._vehicle_speed * 3.6

    @property
    def acceleration_mps2(self):
        return self._vehicle_acceleration

    # Engine properties
    @property
    def throttle_pedal(self):
        return self._throttle * 100

    @throttle_pedal.setter
    def throttle_pedal(self, position_pct):
        if position_pct < 0 or position_pct > 100:
            raise ValueError()
        else:
            self._throttle = position_pct / 100.

    @property
    def engine_torque_nm(self):
        return self._engine_torque

    @property
    def engine_power_kw(self):
        return self._engine_torque * self._engine_speed

    @property
    def engine_speed_rpm(self):
        return self._engine_speed

    # Braking properties
    @property
    def brake_pedal(self):
        return self._brake_pedal * 100

    @brake_pedal.setter
    def brake_pedal(self, position_pct):
        self._brake_pedal = position_pct / 100.0

    @property
    def braking_torque(self):
        return self._brake_torque

    @property
    def braking_force(self):
        return self._braking_force

    # Transmission
    @property
    def gear(self):
        return self._selected_gear

    # Environment
    @property
    def drag_force(self):
        return self._drag_force

    @property
    def rolling_force(self):
        return self._rolling_force

    @property
    def slope_force(self):
        return self._slope_force

    def set_position(self, x=0, y=0, theta=0, wheel_angle=0) -> None:

        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_angle = wheel_angle
        self._vehicle_speed = 0

    def steering(self, speed, wheel_angle_speed, dt):

        self.wheel_angle = self.wheel_angle + wheel_angle_speed * dt

        if self.wheel_angle > 0:
            self.wheel_angle = min(
                self.wheel_angle, self.steering_wheel_angle_max)
        else:
            self.wheel_angle = max(
                self.wheel_angle, -self.steering_wheel_angle_max)

        dx = speed * math.cos(self.theta)
        dy = speed * math.sin(self.theta)
        dtheta = speed * math.tan(self.wheel_angle) / self.wheel_base

        self.x = self.x + dx * dt
        self.y = self.y + dy * dt
        self.theta = self.theta + dtheta * dt

        return self.x, self.y, self.theta

    def _brake_model(self, brake_position) -> float:
        """
        Calculates braking force from brake pedal position.
        @brake_position:
        @return: Braking force
        """
        brake_pressure = np.interp(
            brake_position * 100.0, self.brake_position_curve, self.brake_pressure_position_curve) * 2

        if self._wheel_speed != 0:
            braking_torque_front_dynamic = self.brakes_dynamic_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.brakes_front_pads_count / 4
            braking_torque_rear_dynamic = self.brakes_dynamic_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.brakes_rear_pads_count / 4
            self._braking_torque = 2 * braking_torque_front_dynamic + \
                2 * braking_torque_rear_dynamic
        else:
            braking_torque_front_static = self.brakes_static_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.brakes_front_pads_count / 4

            braking_torque_rear_static = self.brakes_static_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.brakes_rear_pads_count / 4
            self._braking_torque = 2 * braking_torque_front_static + \
                2 * braking_torque_rear_static

        self._braking_force = self._braking_torque / self.tire_dynamic_wheel_radius
        return self._braking_force

    def _engine_model(self, speed: float) -> float:
        """
        Model of the engine.
        @speed: Engine speed
        @return: Engine torque
        """
        self._engine_speed = np.clip(
            speed, self.engine_speed_min, self.engine_speed_max)

        if self._throttle == 0:
            self._engine_torque = - np.interp(
                self._engine_speed, self.engine_speed_curve, self.engine_braking_torque_curve)
        else:
            self._engine_torque = np.interp(
                self._engine_speed, self.engine_speed_curve, self.engine_torque_curve) * self._throttle
        return self._engine_torque

    def _transmission_model(self, engine_torque: float, transmission_speed: float) -> tuple[float, float]:
        """Transmission model
        @engine_torque: Engine torque [Nm]
        @transmission_speed: Transmission speed [rad/s]
        return: tuple(transmission torque, engine speed [rpm])"""
        gear_ratio = self.transmission_gear_ratios[self._shifter_model(
            self._engine_speed) - 1]
        transmission_torque = engine_torque * \
            self.transmission_driveline_efficiency * \
            gear_ratio * self.transmission_final_drive_ratio

        engine_speed = gear_ratio * transmission_speed * \
            30 / math.pi * self.transmission_final_drive_ratio

        return transmission_torque, engine_speed

    def _shifter_model(self, engine_speed: float) -> int:
        """Shifter model. Shifter shifts up when engine speed reaches maximum power and shifts down when engine speed
        is bellow maximum torque speed.
        @engine_speed: Engine speed
        @return: Gear number
        """

        # Upshift
        if engine_speed >= self.engine_max_power_speed and self._selected_gear < self.transmission_gears_number:
            self._selected_gear += 1
        # Downshift
        elif engine_speed <= self.engine_max_torque_speed and self._selected_gear > 1:
            self._selected_gear -= 1

        return self._selected_gear

    def _vehicle_model(self, wheel_torque: float) -> float:
        """Vehicle model
        @wheel_torque: Torque on the wheels/transmission.
        @return: wheel speed [rad/s]
        """
        self._traction_force = wheel_torque / self.tire_wheel_radius
        self._braking_force = self._brake_model(self._brake_pedal)
        self._slope_force = self.vehicle_mass * \
            self.env_gravity_acceleration * math.sin(self.slope)

        if abs(self._vehicle_speed > 0.1):
            self._rolling_force = self.vehicle_mass * self.env_gravity_acceleration * \
                self.env_road_load_coef * math.cos(self.slope)
        else:
            self._rolling_force = 0

        self._drag_force = 0.5 * self.env_air_density * self.vehicle_drag_coef * \
            self.vehicle_front_area * \
            abs(self._vehicle_speed) * self._vehicle_speed

        traction_limit = self.vehicle_mass * self.env_gravity_acceleration * 1.1 * 0.5

        if self._vehicle_speed >= 0:
            self._longitudinal_force = self._traction_force - self._slope_force - \
                self._rolling_force - self._drag_force - self._braking_force
        else:
            self._longitudinal_force = self._traction_force - self._slope_force + \
                self._rolling_force + self._drag_force + self._braking_force

        self._vehicle_acceleration = self._longitudinal_force / self.vehicle_mass
        self._vehicle_speed = self._vehicle_speed + \
            self._vehicle_acceleration * self.simulation_step

        wheel_speed = self._vehicle_speed / self.tire_wheel_radius
        return wheel_speed

    def update(self, dt):

        self.simulation_step = dt
        self.engine_torque = self._engine_model(self._engine_speed)
        self.transmission_torque, self._engine_speed = self._transmission_model(
            self.engine_torque, self._wheel_speed)
        self._wheel_speed = self._vehicle_model(self.transmission_torque)

    def plot_vehicle_telemetry(self, parameters: str) -> None:

        parameters = set(parameters)
        plot_number = 5  # len(parameters)

        if plot_number == 0:
            return

        plot_rows = math.floor(math.sqrt(plot_number))
        plot_cols = math.ceil(plot_number / plot_rows)
        plot_pos = 1

        plt.figure("Vehicle telemetry")

        plt.subplot(plot_rows, plot_cols, plot_pos)
        plt.bar(["Vehicle speed"], self.vehicle_speed_kmph, color="red")
        plt.ylim((0, 250))
        plot_pos += 1

        plt.subplot(plot_rows, plot_cols, plot_pos)
        plt.bar(["Gear"], self.gear, color="green")
        plt.ylim((0, 6))
        plot_pos += 1

        plt.pause(0.1)

    def plot_vehicle_parameters(self, parameters_to_show: str) -> None:

        parameters_to_show = set(parameters_to_show)
        plot_numbers = {"e": 3, "b": 1, "t": 1, "o": 1}

        plot_number = sum(
            plot_numbers[key] for key in plot_numbers if key in parameters_to_show)

        if plot_number == 0:
            return

        plot_rows = math.floor(math.sqrt(plot_number))
        plot_cols = math.ceil(plot_number / plot_rows)
        plot_pos = 1

        plt.figure("Vehicle parameters")
        # Engine
        if "e" in parameters_to_show:
            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Engine power curve")
            plt.xlabel("Engine speed [1/min]")
            plt.ylabel("Engine power [kW]")
            plt.plot(self.engine_speed_curve, self.engine_power_curve / 1e3)
            plt.axvline(self.engine_max_power_speed, color="r", linestyle=":")
            plt.axvspan(self.engine_speed_min,
                        self.engine_speed_max, color="g", alpha=0.2)
            plot_pos += 1

            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Engine torque curve")
            plt.xlabel("Engine speed [1/min]")
            plt.ylabel("Engine torque [Nm]")
            plt.plot(self.engine_speed_curve, self.engine_torque_curve)
            plt.axvline(self.engine_max_torque_speed, color="r", ls=":")
            plot_pos += 1

            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Engine braking curve")
            plt.xlabel("Engine speed [1/min]")
            plt.ylabel("Engine braking torque [Nm]")
            plt.plot(self.engine_speed_curve, self.engine_braking_torque_curve)
            plot_pos += 1

        # Braking
        if "b" in parameters_to_show:
            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Braking")
            plt.xlabel("Brake pedal position [%]")
            plt.ylabel("Brakes pressure [Pa]")
            plt.plot(self.brake_position_curve,
                     self.brake_pressure_position_curve)
            plot_pos += 1
        # Transmission
        if "t" in parameters_to_show:
            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Gear ratios")
            plt.xlabel("Gear index [-]")
            plt.ylabel("Gear ratio [-]")
            plt.bar(range(1, self.transmission_gears_number + 1),
                    self.transmission_gear_ratios)
            plot_pos += 1

        plt.tight_layout()
        plt.show()


def main():

    vehicle = VehicleDynamicModel3dof("mazda.ini")

    interval = np.linspace(0, 100, 1001)

    vehicle_speed = list()
    engine_speed = list()
    engine_power = list()
    gear = list()
    throttle = list()
    brake = list()
    braking_force = list()
    drag_force = list()
    rolling_force = list()
    longitudinal_force = list()
    traction_force = list()

    vehicle.throttle_pedal = 100
    vehicle.brake_pedal = 0

    for t in interval:

        if t > 20:
            # vehicle.throttle_pedal = 0
            pass
        if t > 40:
            #vehicle.throttle_pedal = 50
            pass
        if t > 60:
            vehicle.throttle_pedal = 0
            vehicle.brake_pedal = 100
        if t > 80:
            vehicle.brake_pedal = 100

        vehicle.update(0.1)
        vehicle_speed.append(vehicle.vehicle_speed_kmph)
        engine_speed.append(vehicle.engine_speed_rpm)
        engine_power.append(vehicle.engine_power_kw)
        throttle.append(vehicle.throttle_pedal)
        brake.append(vehicle.brake_pedal)
        gear.append(vehicle.gear)
        braking_force.append(vehicle.braking_force)
        drag_force.append(vehicle.drag_force)
        rolling_force.append(vehicle.rolling_force)
        longitudinal_force.append(vehicle._longitudinal_force)
        traction_force.append(vehicle._traction_force)

    plot_row = 3
    plot_col = 4

    plt.figure("Vehicle")
    plt.subplot(plot_row, plot_col, 1)
    plt.title("Vehicle speed")
    plt.plot(interval, vehicle_speed)

    plt.subplot(plot_row, plot_col, 2)
    plt.title("Engine speed")
    plt.plot(interval, engine_speed)

    plt.subplot(plot_row, plot_col, 3)
    plt.title("Engine power")
    plt.plot(interval, engine_power)

    plt.subplot(plot_row, plot_col, 4)
    plt.title("Gear")
    plt.plot(interval, gear)

    plt.subplot(plot_row, plot_col, 5)
    plt.title("Throttle")
    plt.plot(interval, throttle)

    plt.subplot(plot_row, plot_col, 6)
    plt.title("Brake")
    plt.plot(interval, brake)

    plt.subplot(plot_row, plot_col, 7)
    plt.title("Braking force")
    plt.plot(interval, braking_force)

    plt.subplot(plot_row, plot_col, 8)
    plt.title("Drag force")
    plt.plot(interval, drag_force)

    plt.subplot(plot_row, plot_col, 9)
    plt.title("Rolling force")
    plt.plot(interval, rolling_force)

    plt.subplot(plot_row, plot_col, 10)
    plt.title("Longitudinal force")
    plt.plot(interval, longitudinal_force)

    plt.subplot(plot_row, plot_col, 11)
    plt.title("Traction force")
    plt.plot(interval, traction_force)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vehicle model with 3DOF.')
    parser.add_argument("--parameters")
    parser.add_argument("--telemetry")
    args = parser.parse_args()

    main()
