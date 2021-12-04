import sys
import pytest
# setting path
sys.path.append('../VehicleDynamicModel')

import vehiclemodel3dof



def test_update():

    vm = vehiclemodel3dof.VehicleDynamicModel3dof()
    vm.update()

