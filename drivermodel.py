import numpy as np
import vehiclemodel3dof
import matplotlib.pyplot as plt


class DriverModel(object):

    SLOW_DRIVER = 1
    NORMAL_DRIVER = 0
    FAST_DRIVER = 2

    def __init__(self, driver_type=NORMAL_DRIVER) -> None:
        super().__init__()
        self._vehicle = vehiclemodel3dof.VehicleDynamicModel3dof("mazda.ini")
        self._speed_histeresis = 1
        self._dt = 0.1
        self._driver_type = driver_type
        self._throttle_levels = [80, 50, 100]

        self.vehicle_speeds = list()
        self.target_speed = list()

        self.times = list()
        self.throttle = list()
        self.brake = list()
        self.time = 0

    def wait(self, time):

        t0 = self.time

        while True:

            self._vehicle.update(self._dt)
            self.vehicle_speeds.append(self._vehicle.vehicle_speed_kmph)
            self.throttle.append(self._vehicle.throttle_pedal)

            self.time += self._dt
            self.times.append(self.time)

            if self.time - t0 >= time:
                break



    def drive(self, time, speed, speed_profile=None):

        K_P = 10
        K_I = 4
        K_D = 1

        K_Pb = -2
        K_Ib = -40
        K_Db = -1

        # abs(target_speed - self._vehicle.vehicle_speed_kmph) > self._speed_histeresis:
        while self.time < time:

            if speed_profile is not None:
                if round(self.time) in speed_profile:
                    speed = speed_profile[round(self.time)]

            error = speed - self._vehicle.vehicle_speed_kmph

            prop = K_P * error
            integ = K_I * error * self._dt
            der = K_D * error / self._dt

            prop_b = K_Pb * error
            integ_b = K_Ib * error * self._dt
            der_b = K_Db * error / self._dt

            output = prop + integ + der
            output_b = prop_b + integ_b + der_b

            self._vehicle.throttle_pedal = np.clip(
                output, 0, self._throttle_levels[self._driver_type])

            self._vehicle.brake_pedal = np.clip(
                output_b, 0, self._throttle_levels[self._driver_type])

            self._vehicle.update(self._dt)

            self.vehicle_speeds.append(self._vehicle.vehicle_speed_kmph)
            self.target_speed.append(speed)
            self.throttle.append(self._vehicle.throttle_pedal)
            self.brake.append(self._vehicle.brake_pedal)

            self.time += self._dt
            self.times.append(self.time)



if __name__ == "__main__":

    driver = DriverModel()

    target_speed = 50
    speed_profile = {0: 100, 10: 40, 20: 60}

    driver.drive(40, target_speed, speed_profile)
    

    plt.plot(driver.times, driver.vehicle_speeds, "-", linewidth=4)
    plt.plot(driver.times, driver.throttle)
    plt.plot(driver.times, driver.brake)
    plt.plot(driver.times, driver.target_speed, "--r")
    plt.legend(["Speed", "Throttle", "Brake", "Target speed"])

    plt.show()
