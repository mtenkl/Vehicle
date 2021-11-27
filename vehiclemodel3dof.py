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
            self.wheel_base = body.getfloat("wheelbase", self.wheel_base)
            self.mass = body.getfloat("mass", self.mass)
            self.steering_ratio = body.getfloat("steeringratio", self.steering_ratio)
            self.wheel_angle_max = body.getfloat("maxwheelangle", self.wheel_angle_max)

        

        self.x = 0
        self.y = 0
        self.theta = 0
        self.wheel_angle = 0

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