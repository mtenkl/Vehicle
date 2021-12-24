import sys
import pytest
# setting path
sys.path.append('../VehicleDynamicModel')

import vehiclemodel.vehiclemodel3dof as vehiclemodel3dof

vm = vehiclemodel3dof.VehicleDynamicModel3dof()

def test_update():

    vm.update(0.1)


def test_brake():


    braking_torque = vm._brake_torque(2e6)
    print(braking_torque)