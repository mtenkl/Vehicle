import configparser
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from scipy import interpolate

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
        self.WHEEL_BASE = config.getfloat("vehicle", "wheelbase", fallback=2.5)
        self.VEHICLE_MASS = config.getfloat("vehicle", "mass", fallback=1390)
        self.VEHICLE_FRONT_AREA = config.getfloat(
            "vehicle", "frontArea", fallback=2.4)
        self.VEHICLE_DRAG_COEF = config.getfloat(
            "vehicle", "dragCoeficient", fallback=0.3)

        # Engine parameters
        self.ENGINE_SPEED_MIN = config.getfloat(
            "engine", "engineSpeedMin", fallback=900)
        self.ENGINE_SPEED_MAX = config.getfloat(
            "engine", "engineSpeedMax", fallback=6000)
        self.ENGINE_MAX_TORQUE_SPEED = config.getfloat(
            "engine", "maxTorqueEngineSpeed", fallback=3500)
        self.ENGINE_MAX_POWER_SPEED = config.getfloat(
            "engine", "maxPowerEngineSpeed", fallback=6000)
        self.ENGINE_TORQUE_MIN = config.getfloat(
            "engine", "minTorque", fallback=100)
        self.ENGINE_TORQUE_CURVE = list(map(float, config.get(
            "engine", "torqueAxis", fallback="200 385 439 450 450 367").split()))
        self.ENGINE_SPEED_CURVE = list(map(float, config.get(
            "engine", "engineSpeedAxis", fallback="900 2020 2990 3500 5000 6500").split()))
        self.ENGINE_POWER_CURVE = np.multiply(
            self.ENGINE_TORQUE_CURVE, self.ENGINE_SPEED_CURVE) * np.pi / 30
        self.ENGINE_BRAKING_TORQUE_CURVE = list(map(float, config.get(
            "engine", "brakingTorqueAxis", fallback="30 40 48 45 30 20").split()))
        engine_map_file_name = config.get("engine", "engineMap").strip("\"")
        self.ENGINE_MAP = self._get_engine_map(engine_map_file_name)

        # Steering parameters
        self.STEERING_RATIO = config.getfloat(
            "steering", "steeringRatio", fallback=14)
        self.STEERING_TIRE_ANGLE_MAX = config.getfloat(
            "steering", "maxWheelAngle", fallback=32)

        # Transmission parameters
        self.TRANSMISSION_GEARS_NUMBER = config.getint(
            "transmission", "gears", fallback=6)
        self.TRANSMISSION_GEAR_RATIOS = list(map(float, config.get(
            "transmission", "gearRatios", fallback="-4.71 4.71 3.14 2.11 1.67 1.29 1.00").split()))
        self.TRANSMISSION_FINAL_DRIVE_RATIO = config.getfloat(
            "transmission", "finalDriveRatio", fallback=3)
        self.TRANSMISSION_DRIVELINE_EFFICIENCY = config.getfloat(
            "transmission", "driveLineEfficiency", fallback=0.9)
        self.TRANSMISSION_MODES = {"P": 0, "R": -1, "N": 0, "D": 1}

        # Tire parameters
        self.TIRE_RADIUS = config.getfloat(
            "tire", "wheelRadius", fallback=0.9)
        self.TIRE_DYNAMICR_RADIUS = self.TIRE_RADIUS * 0.98

        # Environment parameters
        self.ENV_AIR_DENSITY = config.getfloat(
            "environment", "airDensity", fallback=1.225)
        self.ENV_ROAD_LOAD_COEF = config.getfloat(
            "environment", "roadLoadCoeficient", fallback=1.225)
        self.ENV_GRAVITY_ACC = config.getfloat(
            "environment", "gravityAcceleration", fallback=9.81)

        # Brakes parameters
        self.BRAKE_FRONT_PADS_NUMBER = config.getfloat(
            "brakes", "frontPads", fallback=2)
        self.BRAKE_REAR_PADS_NUMBER = config.getfloat(
            "brakes", "rearPads", fallback=2)
        self.BRAKE_STATIC_FRICTION = config.getfloat(
            "brakes", "staticFriction", fallback=0.92)
        self.BRAKE_DYNAMIC_FRICTION = config.getfloat(
            "brakes", "dynamicFriction", fallback=0.9)
        self.BRAKE_PISTON_DIAMETER = config.getfloat(
            "brake", "pistonDiameter", fallback=0.02)
        self.BRAKE_PAD_POSITION_RADIUS = config.getfloat(
            "brake", "padPositionRadius", fallback=0.15)
        self.BRAKE_POSITION_CURVE = list(map(float, config.get(
            "brake", "brakePositionPct", fallback="0 5 30 50 70 100").split()))
        self.BRAKE_PRESSURE_POSITION_CURVE = list(map(float, config.get(
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
        self._ignition_on = False
        self._engine_speed = 0
        self._engine_torque = 0
        self._throttle = 0

        # Transmission variables
        self._selected_gear = 1
        self._drive_mode = "N"
        self._drive_mode_ratio = 0

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
    def ignition_on(self):
        return self._ignition_on

    @ignition_on.setter
    def ignition_on(self, value):
        self._ignition_on = value
        if value == True:
            self._engine_speed = self.ENGINE_SPEED_MIN
        else:
            self._engine_speed = 0

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

    @property
    def drive_mode(self):
        return self._drive_mode

    @drive_mode.setter
    def drive_mode(self, value):
        self._drive_mode = value

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

    def _get_engine_map(self, file_path):
        
        engine_map = np.genfromtxt(file_path, dtype="float", delimiter=",")
        throttle_pct = engine_map[0,1:]
        engine_speed = engine_map[1:,0]
        torque = engine_map[1:,1:]

        return interpolate.interp2d(throttle_pct, engine_speed, torque)


    def set_position(self, x=0, y=0, theta=0, wheel_angle=0) -> None:
        """
        Set postion of vehicle
        @x: x coordinates [m]
        @y: y coordinates [m]
        @theta: orientaion of vehicle
        @wheel_angle: orientation of front wheel relatie to vehicle axis
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_angle = wheel_angle
        self._vehicle_speed = 0

    def steering(self, speed, wheel_angle_speed, dt):

        self.wheel_angle = self.wheel_angle + wheel_angle_speed * dt

        self.wheel_angle = np.clip(self.wheel_angle, math.radians(
            -self.STEERING_TIRE_ANGLE_MAX), math.radians(self.STEERING_TIRE_ANGLE_MAX))

        dx = speed * math.cos(self.theta)
        dy = speed * math.sin(self.theta)
        dtheta = speed * math.tan(self.wheel_angle) / self.WHEEL_BASE

        self.x = self.x + dx * dt
        self.y = self.y + dy * dt
        self.theta = self.theta + dtheta * dt

        return self.x, self.y, self.theta

    def _ignition_model(self):
        if self._ignition_on:
            self._engine_speed = self.ENGINE_SPEED_MIN
        else:
            self._engine_speed = 0

    def _brake_model(self, brake_position) -> float:
        """
        Calculates braking force from brake pedal position.
        @brake_position:
        @return: Braking force
        """
        brake_pressure = np.interp(
            brake_position * 100.0, self.BRAKE_POSITION_CURVE, self.BRAKE_PRESSURE_POSITION_CURVE) * 2

        if self._wheel_speed != 0:
            braking_torque_front_dynamic = self.BRAKE_DYNAMIC_FRICTION * brake_pressure * math.pi * self.BRAKE_PISTON_DIAMETER * \
                self.BRAKE_PISTON_DIAMETER * self.BRAKE_PAD_POSITION_RADIUS * \
                self.BRAKE_FRONT_PADS_NUMBER / 4
            braking_torque_rear_dynamic = self.BRAKE_DYNAMIC_FRICTION * brake_pressure * math.pi * self.BRAKE_PISTON_DIAMETER * \
                self.BRAKE_PISTON_DIAMETER * self.BRAKE_PAD_POSITION_RADIUS * \
                self.BRAKE_REAR_PADS_NUMBER / 4
            self._braking_torque = 2 * braking_torque_front_dynamic + \
                2 * braking_torque_rear_dynamic
        else:
            braking_torque_front_static = self.BRAKE_STATIC_FRICTION * brake_pressure * math.pi * self.BRAKE_PISTON_DIAMETER * \
                self.BRAKE_PISTON_DIAMETER * self.BRAKE_PAD_POSITION_RADIUS * \
                self.BRAKE_FRONT_PADS_NUMBER / 4

            braking_torque_rear_static = self.BRAKE_STATIC_FRICTION * brake_pressure * math.pi * self.BRAKE_PISTON_DIAMETER * \
                self.BRAKE_PISTON_DIAMETER * self.BRAKE_PAD_POSITION_RADIUS * \
                self.BRAKE_REAR_PADS_NUMBER / 4
            self._braking_torque = 2 * braking_torque_front_static + \
                2 * braking_torque_rear_static

        self._braking_force = self._braking_torque / self.TIRE_DYNAMICR_RADIUS
        return self._braking_force

    def _engine_model(self, engine_speed: float, throttle: float) -> float:
        """
        Model of the engine.
        @engine_speed: Engine speed [1/min]
        @throttle: Accelerator throttle [0-1]
        @return: Engine torque [Nm]
        """

        engine_torque = self.ENGINE_MAP(throttle * 100, engine_speed)[0]
        return engine_torque

        engine_torque = np.interp(
            engine_speed, self.ENGINE_SPEED_CURVE, self.ENGINE_TORQUE_CURVE) * min(throttle + 0.01, 1)
        return engine_torque

    def _transmission_model(self, engine_torque: float, wheel_speed: float, drive_mode: str) -> tuple[float, float]:
        """
        Transmission model. If drive mode is P or N it returns tuple(0, engine speed)
        @engine_torque: Engine torque [Nm]
        @wheel_speed: Wheel speed [rad/s]
        @drive_mode: Drive mode [P, R, N, D]
        return: tuple(transmission torque [Nm], engine speed [1/min])
        """
        if drive_mode.upper() not in self.TRANSMISSION_MODES.keys():
            raise ValueError(f"Invalid drive mode value '{drive_mode}'")

        if drive_mode.upper() in ["P", "N"]:
            return 0, self._engine_speed
        elif drive_mode.upper() == "D":
            self._selected_gear = self._shifter_model(self._engine_speed)
        else:
            # R gear
            self._selected_gear = 0

        gear_ratio = self.TRANSMISSION_GEAR_RATIOS[self._selected_gear]
                                                   

        transmission_torque = engine_torque * \
            self.TRANSMISSION_DRIVELINE_EFFICIENCY * \
            gear_ratio * self.TRANSMISSION_FINAL_DRIVE_RATIO
        RAD_TO_RPM = 30 / math.pi
        engine_speed = gear_ratio * \
            self.TRANSMISSION_FINAL_DRIVE_RATIO * wheel_speed * RAD_TO_RPM

        return transmission_torque, engine_speed

    def _shifter_model(self, engine_speed: float) -> int:
        """Shifter model. Shifts up when engine speed reaches maximum power and shifts down when engine speed
        is bellow maximum torque speed. Returns gear number request. It do not shift itself.
        @engine_speed: Engine speed
        @return: Gear number
        """
        # Upshift
        if engine_speed >= self.ENGINE_MAX_POWER_SPEED and self._selected_gear < self.TRANSMISSION_GEARS_NUMBER:
            return self._selected_gear + 1
        # Downshift
        elif engine_speed <= self.ENGINE_MAX_TORQUE_SPEED and self._selected_gear > 1:
            return self._selected_gear - 1
        else:
            return self._selected_gear

    def _vehicle_model(self, wheel_torque: float) -> float:
        """Vehicle model
        @wheel_torque: Torque on the wheels/from transmission [Nm]
        @return: wheel speed [rad/s]
        """

        self._traction_force = wheel_torque / self.TIRE_DYNAMICR_RADIUS
        self._braking_force = self._brake_model(self._brake_pedal)
        self._slope_force = -1 * self.VEHICLE_MASS * \
            self.ENV_GRAVITY_ACC * math.sin(self.slope)

        self._rolling_force = self.VEHICLE_MASS * self.ENV_GRAVITY_ACC * \
            self.ENV_ROAD_LOAD_COEF * math.cos(self.slope)

        self._drag_force = 0.5 * self.ENV_AIR_DENSITY * self.VEHICLE_DRAG_COEF * \
            self.VEHICLE_FRONT_AREA * self._vehicle_speed * self._vehicle_speed

        # TODO
        traction_limit_force = self.VEHICLE_MASS * self.ENV_GRAVITY_ACC * 1.1 * 0.5

        passive_force = self._rolling_force + self._drag_force + self._braking_force

        if self._vehicle_speed > 0.1:
            self._longitudinal_force = self._traction_force + \
                self._slope_force - passive_force
        elif self._vehicle_speed < -0.1:
            self._longitudinal_force = self._traction_force + \
                self._slope_force + passive_force
        else:
            self._longitudinal_force = self._traction_force + \
                self._slope_force + math.copysign(self._braking_force * 0 + self._rolling_force, -self._vehicle_speed)

        self._vehicle_acceleration = self._longitudinal_force / self.VEHICLE_MASS
        self._vehicle_speed = self._vehicle_speed + \
            self._vehicle_acceleration * self.simulation_step

        if self._drive_mode == "D":
            self._vehicle_speed = max(0, self._vehicle_speed)
        elif self._drive_mode == "R":
            self._vehicle_speed = min(0, self._vehicle_speed)

        wheel_speed = self._vehicle_speed / self.TIRE_DYNAMICR_RADIUS
        return wheel_speed

    def update(self, dt):
        """
        Updates vehicle model
        @dt: update time step [s]
        """
        self.simulation_step = dt
        self._engine_torque = self._engine_model(
            self._engine_speed, self._throttle)
        self._transmission_torque, engine_speed = self._transmission_model(
            self._engine_torque, self._wheel_speed, self._drive_mode)
        self._engine_speed = np.clip(engine_speed, self.ENGINE_SPEED_MIN, self.ENGINE_SPEED_MAX)
        self._wheel_speed = self._vehicle_model(self._transmission_torque)

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
            plt.plot(self.ENGINE_SPEED_CURVE, self.ENGINE_POWER_CURVE / 1e3)
            plt.axvline(self.ENGINE_MAX_POWER_SPEED, color="r", linestyle=":")
            plt.axvspan(self.ENGINE_SPEED_MIN,
                        self.ENGINE_SPEED_MAX, color="g", alpha=0.2)
            plot_pos += 1

            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Engine torque curve")
            plt.xlabel("Engine speed [1/min]")
            plt.ylabel("Engine torque [Nm]")
            plt.plot(self.ENGINE_SPEED_CURVE, self.ENGINE_TORQUE_CURVE)
            plt.axvline(self.ENGINE_MAX_TORQUE_SPEED, color="r", ls=":")
            plot_pos += 1

            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Engine braking curve")
            plt.xlabel("Engine speed [1/min]")
            plt.ylabel("Engine braking torque [Nm]")
            plt.plot(self.ENGINE_SPEED_CURVE, self.ENGINE_BRAKING_TORQUE_CURVE)
            plot_pos += 1

        # Braking
        if "b" in parameters_to_show:
            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Braking")
            plt.xlabel("Brake pedal position [%]")
            plt.ylabel("Brakes pressure [Pa]")
            plt.plot(self.BRAKE_POSITION_CURVE,
                     self.BRAKE_PRESSURE_POSITION_CURVE)
            plot_pos += 1
        # Transmission
        if "t" in parameters_to_show:
            plt.subplot(plot_rows, plot_cols, plot_pos)
            plt.title("Gear ratios")
            plt.xlabel("Gear index [-]")
            plt.ylabel("Gear ratio [-]")
            plt.bar(range(1, self.TRANSMISSION_GEARS_NUMBER + 1),
                    self.TRANSMISSION_GEAR_RATIOS)
            plot_pos += 1

        plt.tight_layout()
        plt.show()


def main():

    vehicle = VehicleDynamicModel3dof("vehicle/mazda.ini")
    vehicle.ignition_on = True
    vehicle.drive_mode = "D"
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
