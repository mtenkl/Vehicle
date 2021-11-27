import configparser
import math
import matplotlib.pyplot as plt
import logging, sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')



class VehicleDynamicModel3dof():



    def __init__(self, params = None) -> None:

        self.wheel_base = 2.5
        self.mass = 1390
        self.steering_ratio =  14
        self.wheel_angle_max = 0.523

        if params is not None:
            logging.info("Loading {} parameters file.".format(params))
            config = configparser.ConfigParser()
            config.read(params)
            body = config["body"]
            powertrain = config["powertrain"]
            self.wheel_base = body.getfloat("wheelbase", 2.5)
            self.mass = body.getfloat("mass", 1390)
            self.steering_ratio = body.getfloat("steeringratio", 14)
            self.wheel_angle_max = body.getfloat("maxwheelangle", 0.523)

            self.gear_ratios = self._parse_gear_ratios(powertrain, 0)
            self.final_drive_ratio = powertrain.getfloat("finaldrive", 3)
            self.driveline_efficiency = powertrain.getfloat("drivelineefficiency", 0.9)
        

        self.x = 0
        self.y = 0
        self.theta = 0
        self.wheel_angle = 0

    
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


    def update(self, speed, wheel_angle_speed, dt):

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


    def traction_force(self, engine_torque, gear: str, ):

        traction_force = engine_torque * self.gear_ratios[gear] * self.driveline_efficiency


def main():

    vehicle = VehicleDynamicModel3dof("mazda.ini")
    
    x = list()
    y = list()

    for t in range(0, 800):

        wheel_angle_speed_deg = 1

        _x, _y, theta = vehicle.update(2, wheel_angle_speed_deg/180*math.pi, 0.1)
        x.append(_x)
        y.append(_y)
        
        
        
    plt.plot(x,y)
    plt.show()



if __name__ == "__main__":

    main()