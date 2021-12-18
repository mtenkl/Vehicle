import configparser
import math
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
        self.drag_coef = config.getfloat(
            "vehicle", "dragCoeficient", fallback=0.3)

        # Engine parameters
        self.engine_speed_min = config.getfloat(
            "engine", "engineSpeedMin", fallback=900)
        self.engine_speed_max = config.getfloat(
            "engine", "engineSpeedMax", fallback=6000)
        self.max_torque_speed = config.getfloat(
            "engine", "maxTorqueEngineSpeed", fallback=3500)
        self.max_power_speed = config.getfloat(
            "engine", "maxPowerEngineSpeed", fallback=6000)
        self.torque_chart = list(map(float, config.get(
            "engine", "torqueAxis", fallback="306 385 439 450 450 367").split()))
        self.torque_speed = list(map(float, config.get(
            "engine", "engineSpeedAxis", fallback="900 2020 2990 3500 5000 6500").split()))
        self.power_chart = np.multiply(
            self.torque_chart, self.torque_speed) * np.pi / 30
        self.engine_braking_torque_chart = list(map(float, config.get(
            "engine", "brakingTorqueAxis", fallback="30 40 48 45 30 20").split()))

        # Steering parameters
        self.steering_ratio = config.getfloat(
            "steering", "steeringRatio", fallback=14)
        self.wheel_angle_max = config.getfloat(
            "steering", "maxWheelAngle", fallback=32)

        # Transmission parameters
        self.gears_number = config.getfloat(
            "transmission", "gears", fallback=6)
        self.gear_ratios = list(map(float, config.get(
            "transmission", "gearRatios", fallback="4.71 3.14 2.11 1.67 1.29 1.00").split()))
        self.final_drive_ratio = config.getfloat(
            "transmission", "finalDriveRatio", fallback=3)
        self.driveline_efficiency = config.getfloat(
            "transmission", "driveLineEfficiency", fallback=0.9)

        # Tire parameters
        self.wheel_radius = config.getfloat(
            "tire", "wheelRadius", fallback=0.9)
        self.dynamic_wheel_radius = self.wheel_radius * 0.98

        # Environment parameters
        self.air_density = config.getfloat(
            "environment", "airDensity", fallback=1.225)
        self.road_load_coef = config.getfloat(
            "environment", "roadLoadCoeficient", fallback=1.225)
        self.gravity_acceleration = config.getfloat(
            "environment", "gravityAcceleration", fallback=9.81)

        # Brakes parameters
        self.front_brakes_pads_count = config.getfloat(
            "brakes", "frontPads", fallback=2)
        self.rear_brakes_pads_count = config.getfloat(
            "brakes", "rearPads", fallback=2)
        self.brakes_static_friction = config.getfloat(
            "brakes", "staticFriction", fallback=0.92)
        self.brakes_dynamic_friction = config.getfloat(
            "brakes", "dynamicFriction", fallback=0.9)
        self.brake_piston_diameter = config.getfloat(
            "brake", "pistonDiameter", fallback=0.02)
        self.brake_pad_position_radius = config.getfloat(
            "brake", "padPositionRadius", fallback=0.15)
        self.brake_position_pct_map = list(map(float, config.get(
            "brake", "brakePositionPct", fallback="0 5 30 50 70 100").split()))
        self.brake_position_pressure_map = list(map(float, config.get(
            "brake", "brakePositionPressure", fallback="0 0 15e5 30e5 50e5 60e5").split()))

        self.x = 0
        self.y = 0
        self.slope = 0

        # Vehicle variables
        self._vehicle_speed = 0
        self._vehicle_acceleration = 0
        self._wheel_speed = 0

        self._traction_force = 0
        self._total_force = 0

        # Engine variables
        self._engine_speed = 0
        self._engine_torque = 0
        self._throttle = 0

        # Transmission variables
        self._gear = 1

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
        self._brake_position_pct = 0

    @property
    def throttle(self):
        return self._throttle * 100

    @throttle.setter
    def throttle(self, position_pct):
        if position_pct < 0 or position_pct > 100:
            raise ValueError()
        else:
            self._throttle = position_pct / 100.

    @property
    def brake(self):
        return self._brake_position_pct

    @brake.setter
    def brake(self, position_pct):
        self._brake_position_pct = position_pct

    @property
    def braking_torque(self):
        return self._brake_torque

    @property
    def braking_force(self):
        return self._braking_force

    @property
    def gear(self):
        return self._gear

    @property
    def vehicle_speed_mps(self):
        return self._vehicle_speed

    @property
    def vehicle_speed_kmph(self):
        return self._vehicle_speed * 3.6

    @property
    def acceleration_mps2(self):
        return self._vehicle_acceleration

    @property
    def engine_torque_nm(self):
        return self._engine_torque

    @property
    def engine_speed_rpm(self):
        return self._engine_speed

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

    def _brake_torque(self, brake_pressure) -> float:
        """
        Brake model.
        @Returns brakes torque
        """
        if self._wheel_speed != 0:
            braking_torque_front_dynamic = self.brakes_dynamic_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.front_brakes_pads_count / 4

            braking_torque_rear_dynamic = self.brakes_dynamic_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.rear_brakes_pads_count / 4
            self._braking_torque = 2 * braking_torque_front_dynamic + 2 * braking_torque_rear_dynamic
            return self._braking_torque
        else:
            braking_torque_front_static = self.brakes_static_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.front_brakes_pads_count / 4

            braking_torque_rear_static = self.brakes_static_friction * brake_pressure * math.pi * self.brake_piston_diameter * \
                self.brake_piston_diameter * self.brake_pad_position_radius * \
                self.rear_brakes_pads_count / 4
 
            self._braking_torque = 2 * braking_torque_front_static + 2 * braking_torque_rear_static
            return self._braking_torque

    def _brake_force(self, brake_position_pct):
        """
        Calculates braking force from braking torque
        @brake_position_pct:
        @return: Braking force
        """
        brake_pressure = np.interp(brake_position_pct, self.brake_position_pct_map, self.brake_position_pressure_map)
        brake_torque = self._brake_torque(brake_pressure)
        self._braking_force = brake_torque / self.wheel_radius
        return self._braking_force


    def _engine(self, speed: float) -> float:
        """
        Model of the engine.
        @speed: Engine speed
        @return: Engine torque
        """
        self._engine_speed = np.clip(
            speed, self.engine_speed_min, self.engine_speed_max)

        if self._throttle == 0:
            self._engine_torque = - np.interp(
                self._engine_speed, self.torque_speed, self.engine_braking_torque_chart)
        else:
            self._engine_torque = np.interp(
                self._engine_speed, self.torque_speed, self.torque_chart) * self._throttle
        return self._engine_torque

    def _transmission(self, engine_torque: float, transmission_speed: float) -> tuple[float, float]:
        """Transmission model
        @engine_torque: Engine torque [Nm]
        @transmission_speed: Transmission speed [rad/s]
        return: tuple(transmission torque, engine speed [rpm])"""
        gear_ratio = self.gear_ratios[self._shifter(self._engine_speed) - 1]
        transmission_torque = engine_torque * \
            self.driveline_efficiency * gear_ratio * self.final_drive_ratio

        engine_speed = gear_ratio * transmission_speed * \
            30 / math.pi * self.final_drive_ratio

        return transmission_torque, engine_speed

    def _shifter(self, engine_speed: float) -> int:
        """Shifter model. Shifter shifts up when engine speed reaches maximum power and shifts down when engine speed
        is bellow maximum torque speed.
        @engine_speed: Engine speed
        @return: Gear number
        """

        # Upshift
        if engine_speed >= self.max_power_speed and self._gear < self.gears_number:
            self._gear += 1
        # Downshift
        elif engine_speed <= self.max_torque_speed and self._gear > 1:
            self._gear -= 1

        return self._gear

    def _vehicle(self, wheel_torque: float) -> float:
        """Vehicle model
        @wheel_torque: Torque on the wheels/transmission.
        @return: wheel speed [rad/s]
        """
        self._traction_force = wheel_torque / self.wheel_radius

        self._slope_force = self.vehicle_mass * \
            self.gravity_acceleration * math.sin(self.slope)
        if self._vehicle_speed > 0:
            self._rolling_force = self.vehicle_mass * self.gravity_acceleration * \
                self.road_load_coef * math.cos(self.slope)
        elif self._vehicle_speed == 0:
            self._rolling_force = 0
        else:
            self._rolling_force = -1 * self.vehicle_mass * \
                self.gravity_acceleration * \
                self.road_load_coef * math.cos(self.slope)

        self._drag_force = 0.5 * self.air_density * self.drag_coef * \
            self.vehicle_front_area * self._vehicle_speed * self._vehicle_speed

        traction_limit = self.vehicle_mass * self.gravity_acceleration * 1.1 * 0.5

        self._total_force = self._traction_force - \
            (self._slope_force + self._rolling_force + self._drag_force + self._brake_force(self.brake))
        
        
        self._vehicle_acceleration = self._total_force / self.vehicle_mass
        self._vehicle_speed = self._vehicle_speed + \
            self._vehicle_acceleration * self.simulation_step

        wheel_speed = self._vehicle_speed / self.wheel_radius
        return wheel_speed

    def update(self, dt):

        self.simulation_step = dt
        self.engine_torque = self._engine(self._engine_speed)
        self.transmission_torque, self._engine_speed = self._transmission(
            self.engine_torque, self._wheel_speed)
        self._wheel_speed = self._vehicle(self.transmission_torque)


def show_torque(vehicle: VehicleDynamicModel3dof):

    plt.subplot(1, 2, 1)
    plt.plot(vehicle.torque_speed, vehicle.torque_chart)
    plt.xlabel("Engine speed")
    plt.ylabel("Torque")

    plt.subplot(1, 2, 2)
    plt.plot(vehicle.torque_speed, vehicle.power_chart)
    plt.xlabel("Engine speed")
    plt.ylabel("Power")
    plt.show()


def show_vehicle_data(t, telemetry):

    plt.figure("Vehicle")

    plt.subplot(3, 3, 1)
    plt.title("Vehicle speed")
    plt.plot(t, telemetry["vehicle_speed"])

    plt.subplot(3, 3, 2)
    plt.title("Engine speed")
    plt.plot(t, telemetry["engine_speed"])

    plt.subplot(3, 3, 3)
    plt.title("Acceleration")
    plt.plot(t, telemetry["acceleration"])

    plt.subplot(3, 3, 4)
    plt.title("Engine torque")
    plt.plot(t, telemetry["engine_torque"])

    plt.subplot(3, 3, 5)
    plt.title("Gear")
    plt.plot(t, telemetry["gear"])

    plt.subplot(3, 3, 6)
    plt.title("Drag")
    plt.plot(t, telemetry["drag_force"])

    plt.subplot(3, 3, 7)
    plt.title("Total force")
    plt.plot(t, telemetry["total_force"])

    plt.subplot(3, 3, 8)
    plt.title("Throttle")
    plt.plot(t, telemetry["throttle"])

    plt.subplot(3, 3, 9)
    plt.title("Traction force")
    plt.plot(t, telemetry["traction_force"])

    plt.show()


def main():

    vehicle = VehicleDynamicModel3dof("mazda.ini")
    vehicle.throttle = 50
    vehicle.brake = 80
    t = np.linspace(0, 100, 1001)
    f = 0.01
    # show_torque(vehicle)

    v = list()
    telemetry = dict()
    telemetry["vehicle_speed"] = list()
    telemetry["engine_speed"] = list()
    telemetry["acceleration"] = list()
    telemetry["engine_torque"] = list()
    telemetry["gear"] = list()
    telemetry["drag_force"] = list()
    telemetry["total_force"] = list()
    telemetry["traction_force"] = list()
    telemetry["throttle"] = list()

    for i in t:

        vehicle.throttle = 80  # * round(max(0, np.sin(2 * np.pi * f * i)))
        vehicle.update(0.1)
        telemetry["vehicle_speed"].append(vehicle.vehicle_speed_kmph)
        telemetry["engine_speed"].append(vehicle.engine_speed_rpm)
        telemetry["acceleration"].append(vehicle.acceleration_mps2)
        telemetry["engine_torque"].append(vehicle.engine_torque_nm)
        telemetry["gear"].append(vehicle.gear)

        telemetry["drag_force"].append(vehicle.drag_force)
        telemetry["total_force"].append(vehicle._total_force)
        telemetry["traction_force"].append(vehicle._traction_force)
        telemetry["throttle"].append(vehicle.throttle)

    show_vehicle_data(t, telemetry)


if __name__ == "__main__":

    main()
